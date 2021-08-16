import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18, r3d_18
import numpy as np
import matplotlib.pyplot as plt


class r2plus1d_net(nn.Module):
    def __init__(self,num_classes=4):
        super(r2plus1d_net, self).__init__()
        self.fc_hidden1 = 512
        self.fc_hidden2 = 256
        self.num_classes = num_classes

        r2p1d_resnet = r3d_18(pretrained=True)
        modules = list(r2p1d_resnet.children())[:-1]
        self.r2plus1d_resnet = nn.Sequential(*modules)

        self.fc8 = nn.Linear(512, 512)
        self.fc_act = nn.Linear(512, 512)
        self.fc2_act = nn.Linear(512, self.num_classes)

        self.relu = nn.ReLU()

    def forward(self, x_3d):
        # with torch.no_grad():
        x = self.r2plus1d_resnet(x_3d)
        # print(x.shape)
        x = x.view(-1, self.fc_hidden1)
        # print(x.size())

        clip_feats_int = self.relu(self.fc8(x))
        activity_output = self.fc_act(x)
        activity_pred = self.fc2_act(activity_output)

        # print(x.shape)

        clip_feats_int = clip_feats_int.unsqueeze(0)
        clip_feats_int = clip_feats_int.transpose(0, 1)

        activity_output = activity_output.unsqueeze(0)
        activity_output = activity_output.transpose(0, 1)

        return clip_feats_int, activity_pred, activity_output

def get_1x_lr_params(model):
    """
    This generator returns all the parameters for the conv layer of the net.
    """
    b = [model.r2plus1d_resnet]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k


def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc_act, model.fc8, model.fc2_act]
    # b = [model.lstm1]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


if __name__ == "__main__":
    import torch

    inputs = torch.rand(3, 3, 16, 112, 112)
    net = r2plus1d_net()

    outputs = net.forward(inputs)
    print(outputs[0].size(), outputs[1].size(), outputs[2].size())

