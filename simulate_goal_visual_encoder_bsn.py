#!/usr/bin/env python3
"""
使用 BSN 的模拟流水线（不依赖 AllenAct）

说明：
- 该脚本从已采集的 `collected.npz` 读取 RGB 帧（N,H,W,3 uint8），并模拟从
  `ResNetPreprocessor` 到 `ResnetTensorGoalEncoder` 的整个处理流程，使用
  `BSN`

注意：该脚本不再依赖 AllenAct，但仍依赖 `torch` 与 `torchvision`。
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import sys


# Ensure repository root is on sys.path so we can import retina_model without installing allenact
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))


# -------------------- BSN  --------------------
class ReLUX(nn.Module):
    def __init__(self, thre=8):
        super(ReLUX, self).__init__()
        self.thre = thre

    def forward(self, input):
        return torch.clamp(input, 0, self.thre)


relu4 = ReLUX(thre=4)


class multispike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, lens=4):
        ctx.save_for_backward(input)
        ctx.lens = lens
        return torch.floor(relu4(input) + 0.5)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp1 = 0 < input
        temp2 = input < ctx.lens
        return grad_input * temp1.float() * temp2.float(), None


class Multispike(nn.Module):
    def __init__(self, lens=4, spike=multispike):
        super().__init__()
        self.lens = lens
        self.spike = spike

    def forward(self, inputs):
        return self.spike.apply(inputs) / self.lens


class BSN(nn.Module):
    """Basic Spiking Neuron (BSN) - 复制自仓库中实现（neurons_enco.py）。"""

    def __init__(self, T: int, neuron_size, m_weight=0.1):
        super().__init__()
        self.multispike = Multispike()
        self.fc = nn.Linear(T, T)
        self.neuron_m = nn.Parameter(torch.zeros((T, neuron_size)), requires_grad=True)
        self.memory_loss = nn.Parameter(torch.zeros(T, neuron_size), requires_grad=False)
        self.T = T
        self.neuron_size = neuron_size
        self.m_weight = m_weight
        nn.init.constant_(self.fc.bias, 0.5)

    def forward(self, x_seq: torch.Tensor):
        # x_seq expected shape: [B, C] or [B, C, H, W]
        # 内部会增加时间维度并重复到 T，然后计算 multispike，最后返回 time 维度的平均
        x_seq = x_seq.unsqueeze(0)  # [1, B, C, ...]
        x_seq = x_seq.repeat(self.T, 1, 1, *([1] * (x_seq.dim() - 3)))
        t = x_seq.shape[0]
        b = x_seq.shape[1]
        c = x_seq.shape[2]
        simil = self.neuron_m.unsqueeze(1)  # [T,1,neuron_size]

        if len(x_seq.shape) == 5:
            # spatial
            x_seq = self.m_weight * simil.unsqueeze(3).unsqueeze(4).repeat(
                (1, 1, 1, x_seq.shape[-2], x_seq.shape[-1])
            ) + x_seq
        elif len(x_seq.shape) == 3:
            x_seq = self.m_weight * simil + x_seq

        # flatten time & channels for fc
        x_seq_flattened = x_seq.flatten(1)
        h_seq = torch.addmm(self.fc.bias.unsqueeze(1), self.fc.weight, x_seq_flattened)
        spike = self.multispike(h_seq)
        if self.training:
            # update memory_loss (keeps behaviour similar to original)
            try:
                self.memory_loss.data = (
                    self.memory_loss.data
                    + (x_seq + h_seq.view(x_seq.shape)).reshape(t, b, c, -1).mean(1).mean(-1)
                ) / 2
            except Exception:
                pass

        output = spike.view(x_seq.shape)
        # 返回 time 维度平均，恢复到原始输入形状
        return output.mean(0)


# -------------------- ResNet Preprocessor --------------------
from copy import deepcopy
import math
from spikingjelly.activation_based import layer, functional


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return layer.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return layer.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, cnf: str = None, spiking_neuron: callable = None,
                 **kwargs):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.sn1 = spiking_neuron(neuron_size=planes, **deepcopy(kwargs))
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.sn2 = spiking_neuron(neuron_size=planes, **deepcopy(kwargs))
        self.downsample = downsample
        if downsample is not None:
            self.downsample_sn = spiking_neuron(neuron_size=planes, **deepcopy(kwargs))
        self.stride = stride
        self.cnf = cnf

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.sn1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.sn2(out)

        if self.downsample is not None:
            identity = self.downsample_sn(self.downsample(x))

        if self.cnf == 'ADD':
            out = identity + out
        elif self.cnf == 'AND':
            out = identity * out
        elif self.cnf == 'IAND':
            out = identity * (1. - out)
        else:
            raise NotImplementedError

        return out


class SEWResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, cnf: str = None, pe: bool=False, spiking_neuron: callable = None, **kwargs):
        super().__init__()

        # 这里直接复制 Retina 定义依赖
        from tools.retina_model import Retina

        self.retina = Retina(
            image_H=224,
            image_W=224,
            retinal_H=224,
            retinal_W=224,
            retina_field=1,
            em_number=kwargs.get('T', 4),
            sampling_model='gaussian',
        )
        self.pos_embedding = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid(),
        )

        self.T = kwargs.get('T', 4)
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or a 3-element tuple")
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = layer.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.sn1 = spiking_neuron(neuron_size=self.inplanes, **deepcopy(kwargs))
        self.maxpool = layer.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], cnf=cnf, spiking_neuron=spiking_neuron, **kwargs)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], cnf=cnf, spiking_neuron=spiking_neuron,
                                       **kwargs)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], cnf=cnf, spiking_neuron=spiking_neuron,
                                       **kwargs)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], cnf=cnf, spiking_neuron=spiking_neuron,
                                       **kwargs)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(0.1)
        self.fc = layer.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (layer.BatchNorm2d, layer.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                try:
                    if isinstance(m.bn3, layer.BatchNorm2d):
                        nn.init.constant_(m.bn3.weight, 0)
                except Exception:
                    pass

        functional.set_step_mode(self, 'm')

        if pe is None:
            self.conv1.step_mode = 's'
            self.bn1.step_mode = 's'
            self.pe = None
        else:
            self.pe = nn.Parameter(torch.zeros([self.T, 1, 1, 1, 1]))
            torch.nn.init.trunc_normal_(self.pe, std=0.02)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, cnf: str = None,
                    spiking_neuron: callable = None, **kwargs):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, cnf, spiking_neuron, **kwargs))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, cnf=cnf, spiking_neuron=spiking_neuron, **kwargs))

        return nn.Sequential(*layers)

    def _embed(self, x, l_t):
        b, c, d = l_t.shape
        pos_embed = self.pos_embedding(l_t)
        x = x + (pos_embed).unsqueeze(3).unsqueeze(4)
        return x

    def _forward_impl(self, x):
        if self.pe is None:
            x = self.conv1(x)
            x = self.bn1(x)
            x = x
        else:
            x = x + self.pe
            x = self.conv1(x)
            x = self.bn1(x)

        x = self.sn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x):
        x_seq, em = self.retina(x)
        x_seq = self._embed(x_seq, em.to(x.device)).transpose(0,1)
        x_seq[0] = x
        for i in range(1, len(x_seq)):
            x_seq[i] = x_seq[i] + x

        x_seq_final = self._forward_impl(x_seq)

        return x_seq_final


def sew_resnet18(cnf: str = None, spiking_neuron: callable = None, pretrained: str = None, **kwargs):
    model = SEWResNet(BasicBlock, [2, 2, 2, 2], cnf=cnf, spiking_neuron=spiking_neuron, **kwargs)

    if pretrained is not None:
        state_dict = torch.load(pretrained, map_location='cpu')
        model.load_state_dict(state_dict['model'])
    return model


class ResNetEmbedder(nn.Module):
    def __init__(self, resnet, pool=True):
        super().__init__()
        self.model = resnet
        self.pool = pool
        self.eval()

    def forward(self, x):
        with torch.no_grad():
            x = self.model(x)
            if self.pool:
                x = self.model.avgpool(x)
                x = torch.flatten(x, 1)
            x = x.mean(0)
            return x


class ResNetPreprocessorBSN:

    def __init__(
        self,
        input_uuids=None,
        output_uuid="rgb_resnet",
        input_height=224,
        input_width=224,
        output_height=7,
        output_width=7,
        output_dims=512,
        pool=False,
        cnf: str = "ADD",
        spiking_neuron: callable = None,
        spiking_neuron_kwargs: dict = None,
        pretrained: str = None,
        device: torch.device = None,
        device_ids: list = None,
        **kwargs,
    ):
        self.input_uuids = input_uuids or ["rgb"]
        self.output_uuid = output_uuid
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.output_dims = output_dims
        self.pool = pool

        self.cnf = cnf
        self.spiking_neuron = spiking_neuron or BSN
        self.spiking_neuron_kwargs = spiking_neuron_kwargs or {}
        # allow passing T via kwargs (compat with earlier call sites)
        if "T" in kwargs:
            self.spiking_neuron_kwargs.setdefault("T", kwargs["T"])
        self.pretrained = pretrained

        self.device = torch.device("cpu") if device is None else device
        self.device_ids = device_ids or list(range(torch.cuda.device_count()))

        self._resnet = None

    def __call__(self, imgs_or_obs):
        # Maintain previous script behavior: accept numpy array of images
        if isinstance(imgs_or_obs, dict):
            return self.process(imgs_or_obs)
        if isinstance(imgs_or_obs, np.ndarray):
            obs = {self.input_uuids[0]: torch.from_numpy(imgs_or_obs).float().to(self.device)}
            return self.process(obs)
        # fallback: assume tensor
        obs = {self.input_uuids[0]: imgs_or_obs}
        return self.process(obs)

    @property
    def resnet(self) -> ResNetEmbedder:
        if self._resnet is None:
            sew = sew_resnet18(cnf=self.cnf, spiking_neuron=self.spiking_neuron, pretrained=self.pretrained, **self.spiking_neuron_kwargs).to(self.device)
            self._resnet = ResNetEmbedder(sew, pool=self.pool)
        return self._resnet

    def to(self, device: torch.device) -> "ResNetPreprocessorBSN":
        self._resnet = self.resnet
        self._resnet = self._resnet.to(device)
        self.device = device
        return self

    def process(self, obs: dict, *args, **kwargs):
        x = obs[self.input_uuids[0]].to(self.device).permute(0, 3, 1, 2)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.resnet(x)


# -------------------- ResnetTensorGoalEncoder（BSN --------------------
class ResnetTensorGoalEncoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        resnet_tensor_shape: tuple = (512, 7, 7),
        class_dims: int = 32,
        resnet_compressor_hidden_out_dims: tuple = (128, 32),
        combiner_hidden_out_dims: tuple = (128, 32),
        T: int = 4,
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

        # BSN 参数
        self.T = T

        # ResNet 特征压缩器：2 层 1x1 conv + BSN
        self.resnet_compressor = nn.Sequential(
            nn.Conv2d(self.resnet_tensor_shape[0], self.resnet_hid_out_dims[0], 1),
            BSN(T=self.T, neuron_size=self.resnet_hid_out_dims[0]),
            nn.Conv2d(*self.resnet_hid_out_dims[0:2], 1),
            BSN(T=self.T, neuron_size=self.resnet_hid_out_dims[1]),
        )

        # 视觉 + 目标 融合器：2 层 1x1 conv + BSN
        self.target_obs_combiner = nn.Sequential(
            nn.Conv2d(self.resnet_hid_out_dims[1] + self.class_dims, self.combine_hid_out_dims[0], 1),
            BSN(T=self.T, neuron_size=self.combine_hid_out_dims[0]),
            nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
        )

    @property
    def is_blind(self):
        return self.blind

    @property
    def output_dims(self):
        if self.blind:
            return self.class_dims
        else:
            return self.combine_hid_out_dims[-1] * self.resnet_tensor_shape[1] * self.resnet_tensor_shape[2]

    def get_object_type_encoding(self, observations: dict) -> torch.FloatTensor:
        return self.embed_class(observations[self.goal_uuid].to(torch.int64))

    def compress_resnet(self, observations):
        return self.resnet_compressor(observations[self.resnet_uuid])

    def distribute_target(self, observations):
        target_emb = self.embed_class(observations[self.goal_uuid])
        return target_emb.view(-1, self.class_dims, 1, 1).expand(-1, -1, self.resnet_tensor_shape[-2], self.resnet_tensor_shape[-1])

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
        concat_input = torch.cat(embs, dim=1)
        x = self.target_obs_combiner(concat_input)
        x = x.reshape(x.size(0), -1)

        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)


# -------------------- 主流程 --------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_classes", type=int, default=20)
    parser.add_argument("--T", type=int, default=4)
    parser.add_argument(
        "--pretrained",
        default="/data/ssj/robustnav/projects/objectnav_baselines/models/pretrained/checkpoint_max_test_acc1.pth",
        help="SEWResNet 预训练 checkpoint 路径（默认仓库路径），设置为空字符串不加载",
    )
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

    # Create preprocessor using SEWResNet-style pretraining path and T
    pre = ResNetPreprocessorBSN(device=device, pretrained=(args.pretrained if args.pretrained != "" else None), T=args.T)
    encoder = ResnetTensorGoalEncoder(num_classes=args.num_classes, resnet_tensor_shape=(512, 7, 7), class_dims=32, resnet_compressor_hidden_out_dims=(128, 32), combiner_hidden_out_dims=(128, 32), T=args.T)
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
