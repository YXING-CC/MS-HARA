import torch.nn as nn
import torch
from utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional

from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import torch.nn as nn

# __all__ = ['r3d_18', 'ResNet2d', 'resnet18']

model_urls = {
    'r3d_18': 'https://download.pytorch.org/models/r3d_18-b3b3357e.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
}

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock_2d(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock_2d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock_2d only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock_2d")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck_2d(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        pretrained_2d = True
    ):
        super(Bottleneck_2d, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Conv3DSimple(nn.Conv3d):
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        super(Conv3DSimple, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(3, 3, 3),
            stride=stride,
            padding=padding,
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return stride, stride, stride
        
        
class BasicBlock_3d(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock_3d, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
        

class Bottleneck_3d(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):

        super(Bottleneck_3d, self).__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        # Second kernel
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
        
        
class BasicStem(nn.Sequential):
    """The default conv-batchnorm-relu stem
    """
    def __init__(self):
        super(BasicStem, self).__init__(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))


class res_3_2d_net(nn.Module):
    """
    The res_3_2d_net combines resnet3d and resnet2d for model fusion.
    """
    def __init__(self, arch_3d='r3d_18', arch_2d='resnet18', pretrained_3d=True, pretrained_2d=True, progress= True,
                 block_3d=BasicBlock_3d, block_2d=BasicBlock_2d, conv_makers=[Conv3DSimple] * 4,
                 layers=[2, 2, 2, 2], stem=BasicStem, num_classes=400,
                 zero_init_residual_3d=False, zero_init_residual_2d=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None):
                 
        super(res_3_2d_net, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer_2d = norm_layer

        self.inplanes_2d = 64
        self.dilation_2d = 1
        self.inplanes_3d = 64

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.groups_2d = groups
        self.base_width_2d = width_per_group
        self.conv1_2d = nn.Conv2d(3, self.inplanes_2d, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1_2d = norm_layer(self.inplanes_2d)
        self.relu_2d = nn.ReLU(inplace=True)
        self.maxpool_2d = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_2d = self._make_layer_2d(block_2d, 64, layers[0])
        self.layer2_2d = self._make_layer_2d(block_2d, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3_2d = self._make_layer_2d(block_2d, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4_2d = self._make_layer_2d(block_2d, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool_2d = nn.AdaptiveAvgPool2d((1, 1))

        self.stem_3d = stem()
        self.layer1_3d = self._make_layer_3d(block_3d, conv_makers[0], 64, layers[0], stride=1)
        self.layer2_3d = self._make_layer_3d(block_3d, conv_makers[1], 128, layers[1], stride=2)
        self.layer3_3d = self._make_layer_3d(block_3d, conv_makers[2], 256, layers[2], stride=2)
        self.layer4_3d = self._make_layer_3d(block_3d, conv_makers[3], 512, layers[3], stride=2)
        self.avgpool_3d = nn.AdaptiveAvgPool3d((1, 1, 1))

        # 3d-2d fusion conv layers

        # layer1 fusion
        # self.layer1_c1 = nn.Conv3d(in_channels=16, out_channels=4, kernel_size=3, padding=1)
        # self.layer1_c2 = nn.Conv3d(in_channels=4, out_channels=1, kernel_size=3, padding=1)
        self.layer1_c3 = nn.Conv3d(in_channels=16, out_channels=1, kernel_size=3, padding=1)

        # layer2 fusion
        # self.layer2_c1 = nn.Conv3d(in_channels=8, out_channels=4, kernel_size=3, padding=1)
        # self.layer2_c2 = nn.Conv3d(in_channels=4, out_channels=1, kernel_size=3, padding=1)
        self.layer2_c3 = nn.Conv3d(in_channels=8, out_channels=1, kernel_size=3, padding=1)

        # layer3 fusion
        # self.layer3_c1 = nn.Conv3d(in_channels=4, out_channels=2, kernel_size=3, padding=1)
        # self.layer3_c2 = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=3, padding=1)
        self.layer3_c3 = nn.Conv3d(in_channels=4, out_channels=1, kernel_size=3, padding=1)

        # layer4 fusion
        self.layer4_c1 = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=3, padding=1)

        # self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

        # init weights
        self._initialize_weights_3d()

        if zero_init_residual_3d:
            for m in self.modules():
                if isinstance(m, Bottleneck_3d):
                    nn.init.constant_(m.bn3.weight, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual_2d:
            for m in self.modules():
                if isinstance(m, Bottleneck_2d):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock_2d):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

        if pretrained_3d:
            state_dict_3d = load_state_dict_from_url(model_urls[arch_3d], progress=progress)
            keys_org = list(state_dict_3d.keys())
            state_dict_chg_3d = state_dict_3d

            # Generate new key names which equals to the model's key
            new_key = []
            for keys in state_dict_chg_3d:
                split_key = keys.split('.')
                split_key[0] = split_key[0] + '_3d'
                join_key = '.'.join(split_key)
                new_key.append(join_key)

            # update key names for the pretrained model
            for ind in range(len(keys_org)):
                state_dict_chg_3d[new_key[ind]] = state_dict_chg_3d.pop(keys_org[ind])
            # print('pretrained state_dict', len(list(state_dict_chg_3d.keys())), state_dict_chg_3d)

            s_dict = self.state_dict()

            pretrained_dict_3d = {k: v for k, v in state_dict_chg_3d.items() if k in s_dict}
            # print('pretrained_dict_3d', pretrained_dict_3d)
            # print('self.state_dict() before', len(list(self.state_dict().keys())), s_dict)
            s_dict.update(pretrained_dict_3d)
            # print('self.state_dict() after', len(list(self.state_dict().keys())), s_dict)

            self.load_state_dict(s_dict)

        # print('self.state_dict after load', self.state_dict())

        if pretrained_2d:
            state_dict_2d = load_state_dict_from_url(model_urls[arch_2d], progress=progress)
            keys_org_2d = list(state_dict_2d.keys())
            state_dict_chg_2d = state_dict_2d
            # print('state_dict_chg_2d', state_dict_chg_2d)
            # Generate new key names which equals to the model's key
            new_key_2d = []
            for keys in state_dict_chg_2d:
                split_key = keys.split('.')
                split_key[0] = split_key[0] + '_2d'
                join_key = '.'.join(split_key)
                new_key_2d.append(join_key)

            # update key names for the pretrained model
            for ind in range(len(keys_org_2d)):
                state_dict_chg_2d[new_key_2d[ind]] = state_dict_chg_2d.pop(keys_org_2d[ind])

            # for keys in state_dict_chg_2d:
                # print('prestarined model keys:', keys)

            s_dict = self.state_dict()
            # for keys in s_dict:
                # print('model_state_dict:', keys)

            pretrained_dict_2d = {k: v for k, v in state_dict_chg_2d.items() if k in s_dict}
            # print('pretrained_dict_2d', pretrained_dict_2d)
            s_dict.update(pretrained_dict_2d)
            self.load_state_dict(s_dict)

        # print('self.state_dict', self.state_dict())

    def forward(self, x_vid, x_im):
    
        x_vid = self.stem_3d(x_vid)

        x_im = self.conv1_2d(x_im)
        x_im = self.bn1_2d(x_im)
        x_im = self.relu_2d(x_im)
        x_im = self.maxpool_2d(x_im)

        # Layer 1 Fusion
        x_vid = self.layer1_3d(x_vid)
        x_im = self.layer1_2d(x_im)

        # # print('x_vid.size:', x_vid.size(), 'x_im.size:', x_im.size())
        x_vid1 = x_vid.permute((0, 2, 1, 3, 4))
        # x_im1_1 = self.layer1_c1(x_vid1)
        # x_im1_1 = self.layer1_c2(x_im1_1)

        x_im1_2 = self.layer1_c3(x_vid1)
        # x_im1_2 = torch.sigmoid(x_im1_2)

        # x_im1 = x_im1_1 + x_im1_2

        x_im1 = x_im1_2.squeeze()
        # cos_loss = self.cos(x_im, x_im1).abs().mean()
        x_im = x_im*x_im1

        # Layer 2 Fusion
        x_vid = self.layer2_3d(x_vid)
        x_im = self.layer2_2d(x_im)

        # print('x_vid.size:', x_vid.size(), 'x_im.size:', x_im.size())
        x_vid2 = x_vid.permute((0, 2, 1, 3, 4))
        # x_im2_1 = self.layer2_c1(x_vid2)
        # x_im2_1 = self.layer2_c2(x_im2_1)

        x_im2_2 = self.layer2_c3(x_vid2)
        # x_im2_2 = torch.sigmoid(x_im2_2)

        # x_im2 = x_im2_1 + x_im2_2

        x_im2 = x_im2_2.squeeze()
        # cos_loss += self.cos(x_im, x_im2).abs().mean()

        x_im = x_im*x_im2

        # Layer 3 Fusion
        x_vid = self.layer3_3d(x_vid)
        x_im = self.layer3_2d(x_im)

        # print('x_vid.size:', x_vid.size(), 'x_im.size:', x_im.size())
        x_vid3 = x_vid.permute((0, 2, 1, 3, 4))
        # x_im3_1 = self.layer3_c1(x_vid3)
        # x_im3_1 = self.layer3_c2(x_im3_1)

        x_im3_2 = self.layer3_c3(x_vid3)
        # x_im3_2 = torch.sigmoid(x_im3_2)

        # x_im3 = x_im3_1 + x_im3_2

        x_im3 = x_im3_2.squeeze()
        # cos_loss += self.cos(x_im, x_im3).abs().mean()

        x_im = x_im*x_im3

        # Layer 4 Fusion
        x_vid = self.layer4_3d(x_vid)
        x_im = self.layer4_2d(x_im)
        #
        x_vid4 = x_vid.permute((0, 2, 1, 3, 4))
        x_im4_2 = self.layer4_c1(x_vid4)
        # x_im4_2 = torch.sigmoid(x_im4_2)

        x_im4_2_squ = x_im4_2.squeeze()
        # cos_loss += self.cos(x_im, x_im4_2_squ).abs().mean()

        x_im = x_im*x_im4_2_squ

        #
        x_vid = self.avgpool_3d(x_vid)
        x_im = self.avgpool_2d(x_im)

        x_vid = x_vid.flatten(1)
        x_im = x_im.flatten(1)

        x_vid = x_vid.view(x_vid.size(0), -1)
        x_im = x_im.view(x_im.size(0), -1)

        # return x_vid, x_im, -cos_loss
        return x_vid, x_im

    def _make_layer_2d(self, block: Type[Union[BasicBlock_2d, Bottleneck_2d]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False):
        norm_layer = self._norm_layer_2d
        downsample = None
        previous_dilation = self.dilation_2d
        if dilate:
            self.dilation_2d *= stride
            stride = 1
        if stride != 1 or self.inplanes_2d != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes_2d, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes_2d, planes, stride, downsample, self.groups_2d,
                            self.base_width_2d, previous_dilation, norm_layer))
        self.inplanes_2d = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes_2d, planes, groups=self.groups_2d,
                                base_width=self.base_width_2d, dilation=self.dilation_2d,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _make_layer_3d(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes_3d != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes_3d, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes_3d, planes, conv_builder, stride, downsample))

        self.inplanes_3d = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes_3d, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights_3d(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv1_2d, model.bn1_2d, model.layer1_2d, model.layer2_2d, model.layer3_2d, model.layer4_2d,
         model.avgpool_2d, model.stem_3d, model.layer1_3d, model.layer2_3d, model.layer3_3d, model.layer4_3d, model.avgpool_3d]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.layer1_c1, model.layer1_c2, model.layer1_c3, model.layer2_c1, model.layer2_c2, model.layer2_c3,
         model.layer3_c1, model.layer3_c2, model.layer3_c3, model.layer4_c1]
    # b = [model.lstm1]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


if __name__ == "__main__":
    pretrained_3d = True
    pretrained_2d = True
    progress = True

    inputs1 = torch.rand(2, 3, 16, 112, 112)
    inputs2 = torch.rand(2, 3, 224, 224)

    net = res_3_2d_net()
    # print(net)

    # output1, output2, cos_loss = net.forward(inputs1, inputs2)
    output1, output2 = net.forward(inputs1, inputs2)

    # output1 = net.forward(inputs1, inputs2)
    # print('output 3d size:', output1.size(), 'output 2d size:', output2.size(),'cos_loss', cos_loss)
    # print('output 3d size:', output1, 'output 2d size:', output2)
