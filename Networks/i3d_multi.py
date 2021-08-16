import torch.nn as nn
from networks.backbones import resnet3d
import torch
__all__ = ['i3d_resnet18', 'i3d_resnet34', 'i3d_resnet50', 'i3d_resnet101', 'i3d_resnet152']


class I3D(nn.Module):
    """
    Implements a I3D Network for action recognition.

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
        classifier (nn.Module): module that takes the features returned from the
            backbone and returns classification scores.
    """

    def __init__(self, backbone, classifier):
        super(I3D, self).__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        x = self.backbone(x)
        # print('x.size() backbone', x.size())
        clip_feats_int, cls_score, activity_output = self.classifier(x)
        # print('x.size() classifier', x.size())
        return clip_feats_int, cls_score, activity_output


class I3DHead(nn.Module):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
    """

    def __init__(self, num_classes, in_channels, dropout_ratio=0.5):
        super(I3DHead, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout_ratio = dropout_ratio
        # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.relu = nn.ReLU()
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)
        self.fc_act = nn.Linear(self.in_channels, 256)
        self.fc2_act = nn.Linear(self.in_channels, 256)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # print('x classifier.size()', x.size())
        # [N, in_channels]

        cls_score = self.fc_cls(x)

        activity_output = self.fc2_act(x)

        clip_feats_int = self.fc_act(x)

        clip_feats_int = clip_feats_int.unsqueeze(0)
        clip_feats_int = clip_feats_int.transpose(0, 1)

        activity_output = activity_output.unsqueeze(0)
        activity_output = activity_output.transpose(0, 1)

        # print('clip_feats_int.size()', clip_feats_int.size(), 'activity_pred.size()',activity_output.size())
#
        # [N, num_classes]
        return clip_feats_int, cls_score, activity_output


def _load_model(backbone_name, progress, modality, pretrained2d, num_classes, **kwargs):
    backbone = resnet3d(arch=backbone_name, progress=progress, modality=modality, pretrained2d=pretrained2d)
    classifier = I3DHead(num_classes=num_classes, in_channels=2048, **kwargs)
    model = I3D(backbone, classifier)
    return model


def i3d_resnet18(modality='RGB', pretrained2d=True, progress=True, num_classes=21, **kwargs):
    """Constructs a I3D model with a ResNet3d-18 backbone.

    Args:
        modality (str): The modality of input data (RGB or Flow). If 'RGB', the first Conv
            accept a 3-channels input. (Default: RGB)
        pretrained2d (bool): If True, the backbone utilize the pretrained parameters in 2d
            models. (Default: True)
        progress (bool): If True, displays a progress bar of the download to stderr.
            (Default: True)
        num_classes (int): Number of dataset classes. (Default: 21)
    """
    return _load_model('resnet18', progress, modality, pretrained2d, num_classes, **kwargs)


def i3d_resnet34(modality='RGB', pretrained2d=True, progress=True, num_classes=21, **kwargs):
    """Constructs a I3D model with a ResNet3d-34 backbone.

    Args:
        modality (str): The modality of input data (RGB or Flow). If 'RGB', the first Conv
            accept a 3-channels input. (Default: RGB)
        pretrained2d (bool): If True, the backbone utilize the pretrained parameters in 2d
            models. (Default: True)
        progress (bool): If True, displays a progress bar of the download to stderr.
            (Default: True)
        num_classes (int): Number of dataset classes. (Default: 21)
    """
    return _load_model('resnet34', progress, modality, pretrained2d, num_classes, **kwargs)


def i3d_resnet50(modality='RGB', pretrained2d=True, progress=True, num_classes=21, **kwargs):
    """Constructs a I3D model with a ResNet3d-50 backbone.

    Args:
        modality (str): The modality of input data (RGB or Flow). If 'RGB', the first Conv
            accept a 3-channels input. (Default: RGB)
        pretrained2d (bool): If True, the backbone utilize the pretrained parameters in 2d
            models. (Default: True)
        progress (bool): If True, displays a progress bar of the download to stderr.
            (Default: True)
        num_classes (int): Number of dataset classes. (Default: 21)
    """
    return _load_model('resnet50', progress, modality, pretrained2d, num_classes, **kwargs)


def i3d_resnet101(modality='RGB', pretrained2d=True, progress=True, num_classes=21, **kwargs):
    """Constructs a I3D model with a ResNet3d-101 backbone.

    Args:
        modality (str): The modality of input data (RGB or Flow). If 'RGB', the first Conv
            accept a 3-channels input. (Default: RGB)
        pretrained2d (bool): If True, the backbone utilize the pretrained parameters in 2d
            models. (Default: True)
        progress (bool): If True, displays a progress bar of the download to stderr.
            (Default: True)
        num_classes (int): Number of dataset classes. (Default: 21)
    """
    return _load_model('resnet101', progress, modality, pretrained2d, num_classes, **kwargs)


def i3d_resnet152(modality='RGB', pretrained2d=True, progress=True, num_classes=21, **kwargs):
    """Constructs a I3D model with a ResNet3d-152 backbone.

    Args:
        modality (str): The modality of input data (RGB or Flow). If 'RGB', the first Conv
            accept a 3-channels input. (Default: RGB)
        pretrained2d (bool): If True, the backbone utilize the pretrained parameters in 2d
            models. (Default: True)
        progress (bool): If True, displays a progress bar of the download to stderr.
            (Default: True)
        num_classes (int): Number of dataset classes. (Default: 21)
    """
    return _load_model('resnet152', progress, modality, pretrained2d, num_classes, **kwargs)

if __name__ == "__main__":
    arch = 'i3d_resnet50'
    modality = 'RGB'
    inputs1 = torch.rand(2, 3, 16, 112, 112)
    model = i3d_resnet50(modality=modality, num_classes=5, dropout_ratio=0.5)
    clip_feats_int, cls_score, activity_output = model.forward(inputs1)

    print('output', clip_feats_int.size())
    # encoder_model = getattr(i3d_resnet50, arch)(modality=modality, num_classes=5, dropout_ratio=0.5)