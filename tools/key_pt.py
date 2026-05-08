import torch
from collections import OrderedDict

CHECKPOINT_PATH = "/data/ssj/robustnav/projects/objectnav_baselines/data_collect/bsn.pt"


def print_all_keys(obj, prefix=""):
    """递归打印所有嵌套的键名 + 对应值"""
    if isinstance(obj, (dict, OrderedDict)):
        for key in obj.keys():
            full_key = f"{prefix}.{key}" if prefix else key
            value = obj[key]
            
            # 打印键
            print(f"[键] {full_key}")
            
            # 打印值
            if isinstance(value, torch.Tensor):
                print(f"[值] 张量，形状: {value.shape}，数据类型: {value.dtype}")
                # 如果张量很小，直接打印内容
                if value.numel() <= 100:
                    print(f"[内容] {value}")
            elif isinstance(value, (list, tuple)):
                print(f"[值] 列表/元组，长度: {len(value)}")
                if len(value) <= 20:
                    print(f"[内容] {value}")
            else:
                print(f"[值] {value}")
            
            print("-" * 60) 
            
            # 递归遍历子字典
            print_all_keys(value, prefix=full_key)


if __name__ == "__main__":
    print(f"正在加载文件：{CHECKPOINT_PATH}")
    # 加载 checkpoint（CPU 模式兼容，避免 CUDA 错误）
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    print("\n===== 所有键 + 对应值 =====")
    print_all_keys(checkpoint)