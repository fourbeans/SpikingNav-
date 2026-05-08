#!/usr/bin/env python3
"""
ANN 模拟流水线（不依赖 AllenAct）

说明：
- 从已采集的 `.npz` 读取 RGB 帧（N,H,W,3 uint8），并模拟从
  `ResNetPreprocessor` 到 `ResnetTensorGoalEncoder` 的处理流程，使用 torchvision 的
  `resnet18`（不依赖 AllenAct）。


注意：不包含 BSN/SEW/SpikingJelly ，使用 ANN（torchvision.resnet18）。
"""
import argparse
import os
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
from torchvision import models


class ResNetEmbedder(nn.Module):
    """Small wrapper that runs the ResNet conv layers and optionally pools.

    Implementation follows allenact.embodiedai.preprocessors.resnet.ResNetEmbedder.
    """

    def __init__(self, resnet: models.ResNet, pool: bool = True):
        super().__init__()
        self.model = resnet
        self.pool = pool
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)

            x = self.model.layer1(x)
            x = self.model.layer2(x)
            x = self.model.layer3(x)
            x = self.model.layer4(x)

            if not self.pool:
                return x
            x = self.model.avgpool(x)
            x = torch.flatten(x, 1)
            return x


class ResNetPreprocessor:
    """
    - 使用 torchvision 的 ResNet（默认 resnet18）。
    - 在 `pool=False` 时返回 `[B, 512, 7, 7]` 特征图。
    """

    def __init__(
        self,
        input_uuids: Optional[List[str]] = None,
        output_uuid: str = "rgb_resnet",
        input_height: int = 224,
        input_width: int = 224,
        output_height: int = 7,
        output_width: int = 7,
        output_dims: int = 512,
        pool: bool = False,
        torchvision_resnet_model=models.resnet18,
        pretrained: bool = True,
        device: Optional[torch.device] = None,
        device_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        self.input_uuids = input_uuids or ["rgb"]
        self.output_uuid = output_uuid
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.output_dims = output_dims
        self.pool = pool

        self.make_model = torchvision_resnet_model
        self.pretrained = pretrained

        self.device = torch.device("cpu") if device is None else device
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))

        self._resnet: Optional[ResNetEmbedder] = None

    @property
    def resnet(self) -> ResNetEmbedder:
        if self._resnet is None:
            model = self.make_model(pretrained=self.pretrained).to(self.device)
            self._resnet = ResNetEmbedder(model, pool=self.pool)
        return self._resnet

    def to(self, device: torch.device) -> "ResNetPreprocessor":
        self._resnet = self.resnet.to(device)
        self.device = device
        return self

    def __call__(self, imgs_or_obs):
        # 支持三种调用方式：dict, numpy array, tensor
        if isinstance(imgs_or_obs, dict):
            return self.process(imgs_or_obs)
        if isinstance(imgs_or_obs, np.ndarray):
            obs = {self.input_uuids[0]: torch.from_numpy(imgs_or_obs).float().to(self.device)}
            return self.process(obs)
        obs = {self.input_uuids[0]: imgs_or_obs}
        return self.process(obs)

    def process(self, obs: dict, *args, **kwargs) -> torch.Tensor:
        x = obs[self.input_uuids[0]].to(self.device).permute(0, 3, 1, 2)  # bhwc -> bchw
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.resnet(x)


class ResnetTensorGoalEncoder(nn.Module):
    """
    接受从 ResNetPreprocessor 输出的特征（`rgb_resnet`），融合目标 embedding，输出扁平化向量。
    """

    def __init__(
        self,
        num_classes: int,
        resnet_tensor_shape: tuple = (512, 7, 7),
        class_dims: int = 32,
        resnet_compressor_hidden_out_dims: tuple = (128, 32),
        combiner_hidden_out_dims: tuple = (128, 32),
    ) -> None:
        super().__init__()
        self.goal_uuid = "goal"
        self.resnet_uuid = "rgb_resnet"
        self.class_dims = class_dims

        self.resnet_hid_out_dims = resnet_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims

        self.embed_class = nn.Embedding(num_embeddings=num_classes, embedding_dim=self.class_dims)

        self.blind = False

        self.resnet_tensor_shape = resnet_tensor_shape

        # ResNet 特征压缩器（ANN）：两层 1x1 conv + ReLU
        self.resnet_compressor = nn.Sequential(
            nn.Conv2d(self.resnet_tensor_shape[0], self.resnet_hid_out_dims[0], 1),
            nn.ReLU(),
            nn.Conv2d(self.resnet_hid_out_dims[0], self.resnet_hid_out_dims[1], 1),
            nn.ReLU(),
        )

        # 视觉 + 目标 融合器（ANN）：两层 1x1 conv (+ ReLU)
        self.target_obs_combiner = nn.Sequential(
            nn.Conv2d(self.resnet_hid_out_dims[1] + self.class_dims, self.combine_hid_out_dims[0], 1),
            nn.ReLU(),
            nn.Conv2d(self.combine_hid_out_dims[0], self.combine_hid_out_dims[1], 1),
        )

    @property
    def is_blind(self):
        return self.blind

    @property
    def output_dims(self):
        if self.blind:
            return self.class_dims
        return self.combine_hid_out_dims[-1] * self.resnet_tensor_shape[1] * self.resnet_tensor_shape[2]

    def get_object_type_encoding(self, observations: dict) -> torch.FloatTensor:
        return self.embed_class(observations[self.goal_uuid].to(torch.int64))

    def compress_resnet(self, observations):
        return self.resnet_compressor(observations[self.resnet_uuid])

    def distribute_target(self, observations):
        target_emb = self.embed_class(observations[self.goal_uuid])
        return target_emb.view(-1, self.class_dims, 1, 1).expand(
            -1, -1, self.resnet_tensor_shape[-2], self.resnet_tensor_shape[-1]
        )

    def adapt_input(self, observations):
        resnet = observations[self.resnet_uuid]

        use_agent = False
        nagent = 1

        if len(resnet.shape) == 6:
            use_agent = True
            nstep, nsampler, nagent = resnet.shape[:3]
        else:
            nstep, nsampler = resnet.shape[:2]

        observations[self.resnet_uuid] = resnet.view(-1, *resnet.shape[-3:])
        observations[self.goal_uuid] = observations[self.goal_uuid].view(-1, 1)

        return observations, use_agent, nstep, nsampler, nagent

    @staticmethod
    def adapt_output(x, use_agent, nstep, nsampler, nagent):
        if use_agent:
            return x.view(nstep, nsampler, nagent, -1)
        return x.view(nstep, nsampler * nagent, -1)

    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(observations)

        if self.blind:
            return self.embed_class(observations[self.goal_uuid])

        embs = [self.compress_resnet(observations), self.distribute_target(observations)]
        x = self.target_obs_combiner(torch.cat(embs, dim=1))
        x = x.reshape(x.size(0), -1)

        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--no_pretrained", action="store_true", help="不要加载 torchvision 的 ImageNet 预训练权重")
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    images = data["images"]
    goals = data["goals"] if "goals" in data.files else None
    target_names = data["target_names"] if "target_names" in data.files else None

    N = int(images.shape[0])
    device = torch.device(args.device)

    if goals is None or len(goals) == 0:
        print("未检测到 goals，随机生成")
        rng = np.random.RandomState(12345)
        goals = rng.randint(0, args.num_classes, size=(N,)).astype(np.int64)

    if target_names is None:
        target_names = np.array([str(i) for i in range(args.num_classes)], dtype=object)

    pretrained_flag = not getattr(args, "no_pretrained", False)
    pre = ResNetPreprocessor(
        input_uuids=["rgb"],
        torchvision_resnet_model=models.resnet18,
        pretrained=pretrained_flag,
        pool=False,
        device=device,
    )

    encoder = ResnetTensorGoalEncoder(
        num_classes=args.num_classes,
        resnet_tensor_shape=(512, 7, 7),
        class_dims=32,
        resnet_compressor_hidden_out_dims=(128, 32),
        combiner_hidden_out_dims=(128, 32),
    )
    encoder.to(device)
    encoder.eval()

    batch_size = int(args.batch_size)
    outputs_batches = []

    for i in range(0, N, batch_size):
        batch_imgs = images[i : i + batch_size]
        batch_goals = goals[i : i + batch_size]

        with torch.no_grad():
            feat = pre(batch_imgs)  # [B,512,7,7]

        # prepare observations expected by encoder: [nstep, nsampler, C, H, W]
        # we use nstep=1, nsampler=B
        resnet_obs = feat.unsqueeze(0)  # [1,B,512,7,7]
        goal_t = torch.from_numpy(np.asarray(batch_goals)).to(device).view(-1)

        obs = {"rgb_resnet": resnet_obs.to(device), "goal": goal_t.to(device)}

        with torch.no_grad():
            x_batch = encoder(obs)  # shape [1, B, D]

        # convert to per-sample shape [B,1,1,D]
        x_squeezed = x_batch.squeeze(0).cpu()  # [B,D]
        B = x_squeezed.shape[0]
        D = x_squeezed.shape[1]
        x_per_sample = x_squeezed.view(B, 1, 1, D)
        outputs_batches.append(x_per_sample)

    xs_stacked = torch.cat(outputs_batches, dim=0)  # (N,1,1,D)

    save_dict = {
        "x_raw": xs_stacked,
        "goals": torch.from_numpy(np.asarray(goals, dtype=np.int64)),
        "target_names": np.array(target_names, dtype=object),
    }

    out_dir = os.path.dirname(args.out)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    torch.save(save_dict, args.out)
    print(f"完成：保存到 {args.out}，x_raw shape={xs_stacked.shape}")


if __name__ == "__main__":
    main()
