import numpy as np

# NPZ_PATH = "/data/ssj/robustnav/projects/objectnav_baselines/data_collect/collected_test.npz"
NPZ_PATH = "/data/ssj/robustnav/projects/objectnav_baselines/data_collect/collected_128.npz"


# 加载文件
data = np.load(NPZ_PATH, allow_pickle=True)

print("="*50)
print(f"加载文件: {NPZ_PATH}")
print(f"文件包含所有字段: {list(data.files)}")
print("="*50)

# 逐个打印关键字段
for key in ["images", "goals", "target_names", "scenes"]:
    if key not in data:
        print(f"\n {key}: 不存在")
        continue
    
    arr = data[key]
    print(f"\n{key}:")
    print(f"   形状 (shape): {arr.shape}")
    print(f"   数据类型 (dtype): {arr.dtype}")
    
    if key == "images":
        print(f"   格式: HWC = {arr.shape[1:]}，单帧像素值范围: {arr.min()} ~ {arr.max()}")
        print(f"   前2帧预览: shape = {arr[:2].shape}")
    elif key in ["goals"]:
        print(f"   数值内容: {arr}")
    elif key in ["target_names", "scenes"]:
        print(f"   内容: {arr}")

print("\n" + "="*50)
data.close()