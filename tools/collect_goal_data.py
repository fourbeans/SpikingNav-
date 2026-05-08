#!/usr/bin/env python3
"""
采集脚本（从 RoboTHOR / AI2-THOR 中采集若干完整任务帧，包含 RGB 图像与任务目标）。

说明：
- 脚本会尽量使用 `ai2thor.controller.Controller`。如果环境中安装并配置了 AllenAct / RoboTHOR，这个脚本可以直接运行。
- 脚本会保存一个 `.npz` 文件，包含 `images`、`goals`、`scenes`、`target_names` 等字段。

使用示例：
python collect_goal_data.py --n_episodes 16 --frames_per_episode 8 --out  /data/collected_128.npz --scenes FloorPlan_Train1_1 FloorPlan_Train1_2 FloorPlan_Train1_3 FloorPlan_Train2_1 --seed 12345
"""
import argparse
import os
import sys
import time
import random
import json
import numpy as np


def try_import_ai2thor():
    try:
        from ai2thor.controller import Controller

        return Controller
    except Exception:
        return None


def collect_from_ai2thor(controller_cls, args):
    from PIL import Image

    ctrl = None
    # RNG for reproducible choices inside collection
    rng = random.Random(args.seed) if getattr(args, "seed", None) is not None else random
    try:
        # 尝试使用常见参数启动 Controller
        ctrl = controller_cls(width=args.width, height=args.height, agentMode="locobot", commit_id=args.commit_id) if args.commit_id else controller_cls(width=args.width, height=args.height, agentMode="locobot")
    except Exception:
        try:
            ctrl = controller_cls()
        except Exception as e:
            raise RuntimeError(f"无法启动 ai2thor Controller: {e}")

    collected_images = []
    collected_goals = []
    collected_scenes = []
    target_set = []

    scenes = args.scenes if args.scenes else [args.default_scene]

    print(f"开始采集：episodes={args.n_episodes}, frames/ep={args.frames_per_episode}, scenes={scenes}")

    for ep in range(args.n_episodes):
        # 按需求随机选择场景或按顺序轮转
        if getattr(args, "randomize_scenes", False) and scenes:
            scene = rng.choice(scenes)
        else:
            scene = scenes[ep % len(scenes)]
        print(f"采集 episode {ep+1}/{args.n_episodes} -> scene={scene}")
        # reset 场景
        try:
            event = ctrl.reset(scene)
        except Exception:
            try:
                event = ctrl.reset(scene.replace("FloorPlan_", "FloorPlan_Train"))
            except Exception as e:
                print(f"无法 reset 场景 {scene}: {e}")
                continue

        # 初始化
        try:
            ctrl.step(action="Initialize")
        except Exception:
            try:
                ctrl.step(dict(action="Initialize"))
            except Exception:
                pass

        # 先获取场景内对象列表，从中挑一个作为目标类型（objectType）
        # 使用 reset 返回的 event 或 controller.last_event 获取元数据，避免调用不存在的动作
        try:
            evt = event if event is not None else getattr(ctrl, "last_event", None)
        except Exception:
            evt = getattr(ctrl, "last_event", None)
        objs = []
        try:
            # 常见元数据字段：event.metadata['objects'] 或 event.metadata.get('objects', [])
            objs = evt.metadata.get("objects", []) if evt is not None else []
        except Exception:
            objs = []

        # 如果没有 objects 信息则尝试再次获取或者跳过
        if not objs:
            print("未在元数据中发现 objects 信息，尝试在场景中移动以触发对象列表…")
            try:
                # 尝试多种调用方式以兼容不同 ai2thor 版本
                try:
                    ctrl.step(action="RandomizeObjects")
                except Exception:
                    try:
                        ctrl.step("RandomizeObjects")
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                # 从 controller 的 last_event 获取最新元数据，或在必要时发一个轻量 step("Pass") 来刷新
                evt = getattr(ctrl, "last_event", None)
                if evt is None:
                    try:
                        evt = ctrl.step(action="Pass")
                    except Exception:
                        try:
                            evt = ctrl.step("Pass")
                        except Exception:
                            evt = None
                objs = evt.metadata.get("objects", []) if evt is not None else []
            except Exception:
                objs = []

        if not objs:
            print(f"场景 {scene} 未找到对象列表，episode 跳过")
            continue

        # 选择一个目标对象类型（默认每个 episode 固定目标；若 --goal_per_frame 则在每帧重新选择）
        if not getattr(args, "goal_per_frame", False):
            chosen_obj = rng.choice(objs)
            goal_type = chosen_obj.get("objectType", "UNKNOWN")
            if goal_type not in target_set:
                target_set.append(goal_type)
            goal_idx = target_set.index(goal_type)

        # 获取可达位置并在这些位置上抓若干帧
        try:
            rp_event = ctrl.step(action="GetReachablePositions")
        except Exception:
            try:
                rp_event = ctrl.step("GetReachablePositions")
            except Exception:
                rp_event = None

        reachable = []
        if rp_event is not None:
            # 兼容不同版本返回字段
            reachable = rp_event.metadata.get("reachablePositions", []) or rp_event.metadata.get("actionReturn", [])

        if not reachable:
            print("未获取到 reachablePositions，尝试在原位采集")

        frames_captured = 0
        attempts = 0
        max_attempts = max(50, args.frames_per_episode * 10)

        while frames_captured < args.frames_per_episode and attempts < max_attempts:
            attempts += 1
            try:
                if reachable:
                    pos = rng.choice(reachable)
                    # Teleport 到位置（不同版本接口可能不同），优先使用 keyword 形式
                    try:
                        ctrl.step(action="TeleportFull", x=pos.get("x", 0), y=pos.get("y", 0), z=pos.get("z", 0), rotation=pos.get("rotation", 0))
                    except Exception:
                        try:
                            ctrl.step(action="Teleport", x=pos.get("x", 0), y=pos.get("y", 0), z=pos.get("z", 0))
                        except Exception:
                            try:
                                ctrl.step("TeleportFull", x=pos.get("x", 0), y=pos.get("y", 0), z=pos.get("z", 0), rotation=pos.get("rotation", 0))
                            except Exception:
                                try:
                                    ctrl.step("Teleport", x=pos.get("x", 0), y=pos.get("y", 0), z=pos.get("z", 0))
                                except Exception:
                                    pass
                # 获取当前帧：优先使用 controller.last_event（reset/teleport 会更新），否则发一个 "Pass" 步骤刷新
                try:
                    evt = getattr(ctrl, "last_event", None)
                except Exception:
                    evt = None
                if evt is None:
                    try:
                        evt = ctrl.step(action="Pass")
                    except Exception:
                        try:
                            evt = ctrl.step("Pass")
                        except Exception:
                            evt = None
                if evt is None:
                    continue
                frame = None
                try:
                    frame = evt.frame
                except Exception:
                    # 有些版本可能使用 evt.return_data 或 evt.event.frame
                    try:
                        frame = evt.cv2_image
                    except Exception:
                        frame = None

                if frame is None:
                    # 尝试通过 metadata 中的 rgb 字段
                    try:
                        frame = evt.metadata.get("last_event", {}).get("frame")
                    except Exception:
                        frame = None

                if frame is None:
                    time.sleep(0.1)
                    continue

                # 转为 HWC uint8
                if isinstance(frame, np.ndarray):
                    img = frame
                else:
                    # 试用 PIL Image 的接口
                    try:
                        img = np.asarray(frame)
                    except Exception:
                        continue

                # 如果设置了 --goal_per_frame，则为本帧重新选择目标
                if getattr(args, "goal_per_frame", False):
                    chosen_obj = rng.choice(objs)
                    goal_type = chosen_obj.get("objectType", "UNKNOWN")
                    if goal_type not in target_set:
                        target_set.append(goal_type)
                    goal_idx = target_set.index(goal_type)

                collected_images.append(img.astype(np.uint8))
                collected_goals.append(int(goal_idx))
                collected_scenes.append(scene)
                frames_captured += 1
                print(f"采集帧 {frames_captured}/{args.frames_per_episode} (episode {ep+1})")
            except Exception as e:
                print(f"采集帧时出错：{e}")
                time.sleep(0.05)
                continue

    # 保存为 npz
    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    np.savez_compressed(
        args.out,
        images=np.stack(collected_images) if len(collected_images) > 0 else np.zeros((0, args.height, args.width, 3), dtype=np.uint8),
        goals=np.array(collected_goals, dtype=np.int32),
        scenes=np.array(collected_scenes, dtype=object),
        target_names=np.array(target_set, dtype=object),
    )

    print(f"采集完成，保存到 {args.out}，共 {len(collected_images)} 帧，{len(target_set)} 个目标类型。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument("--frames_per_episode", type=int, default=8)
    parser.add_argument("--total_frames", type=int, default=None, help="采集的总帧数；若设置则覆盖 n_episodes/frames_per_episode（等于 total_frames 个 episode，每个 episode 1 帧）")
    parser.add_argument("--goal_per_frame", action="store_true", help="每帧随机选择目标对象以提高目标类型多样性（默认每个 episode 固定目标）")
    parser.add_argument("--randomize_scenes", action="store_true", help="如果传入多个 --scenes，则每个 episode 随机选择场景（增加场景多样性）")
    parser.add_argument("--seed", type=int, default=None, help="随机种子，便于可重复采集")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--default_scene", type=str, default="FloorPlan_Train1_1")
    parser.add_argument("--commit_id", type=str, default=None)

    args = parser.parse_args()

    # If total_frames provided, override n_episodes and frames_per_episode
    if args.total_frames is not None:
        args.n_episodes = int(args.total_frames)
        args.frames_per_episode = 1

    if args.seed is not None:
        random.seed(int(args.seed))

    Controller = try_import_ai2thor()
    if Controller is None:
        print("未检测到 ai2thor，脚本无法直接从模拟器采集。请在已安装 ai2thor 的环境中运行此脚本。")
        sys.exit(1)

    collect_from_ai2thor(Controller, args)


if __name__ == "__main__":
    main()
