# 操作指南

## 一、脚本功能概述
| 脚本文件 | 核心功能 |
|----------|----------|
| `simulate_goal_visual_encoder_ann.py` | 无BSN/SEW/SpikingJelly依赖，采用torchvision原生ResNet18实现视觉特征提取与目标嵌入融合，是ANN架构下视觉编码任务的核心执行入口 |
| `simulate_goal_visual_encoder_bsn.py` | 依赖SpikingJelly实现脉冲神经元计算，复刻仓库中BSN神经元逻辑，对应脉冲神经网络版本的视觉编码仿真，与ANN版本形成对照实验 |

## 二、脚本运行方法

### ANN版本脚本运行
```bash
python simulate_goal_visual_encoder_ann.py --input data/collected_128.npz --out output/encoded_x_128frame_ann.pt --batch_size 64 --device cuda:0

### BSN版本脚本运行
```bash
python simulate_goal_visual_encoder_bsn.py --input data/collected_128.npz --out output/encoded_x_128frame_bsn.pt --batch_size 64 --device cuda:0
```

## 三、包依赖清单
### 基础依赖（两个脚本均需）
| 包名 | 最低版本 | 说明 |
|------|----------|------|
| python | 3.7 | 脚本运行基础环境 |
| torch | 1.9.0 | 核心张量计算框架 |
| torchvision | 0.10.0 | ANN版本ResNet18依赖，提供图像预处理/模型加载 |
| numpy | 1.21.0 | 数据读取（npz文件）与数组处理 |
| argparse | - | Python内置，参数解析（无需额外安装） |

### BSN脚本额外依赖（仅`simulate_goal_visual_encoder_bsn.py`需要）
| 包名 | 最低版本 | 说明 |
|------|----------|------|
| spikingjelly | 0.9.0 | 脉冲神经网络（SNN）核心计算库 |

### 快速安装
```bash
# 安装基础依赖
pip install torch>=1.9.0 torchvision>=0.10.0 numpy>=1.21.0

# 运行BSN版本，额外安装
pip install spikingjelly>=0.9.0
```

### 补充说明
- 若需GPU加速，需安装与本地CUDA版本匹配的`torch`/`torchvision`；
- `retina_model` 为仓库内部模块，脚本已内置路径处理逻辑，无需额外安装；
- 内置模块（os/copy/math/typing）为Python标准库，无需单独安装。
