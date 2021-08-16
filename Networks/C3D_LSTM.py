import torch
import torch.nn as nn
from Cardata_path import Path

class C3D_Encoder(nn.Module):
    """
    The C3D network as described in [1].
    """

    def __init__(self, pretrained=False, num_classes=4):
        super(C3D_Encoder, self).__init__()

        print('this is c3d encoder')

        self.clip_len = 16
        self.lstm_hidden_size = 60

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.num_classes = num_classes
        self.fc2_act = nn.Linear(256, self.num_classes)

        self.__init_weight()

        # self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 256)
        self.fc_act = nn.Linear(4096, 256)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):

        batch, c, frames, H, W = x.shape
        # with torch.no_grad():
        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, 8192)
        h = self.relu(self.fc6(h))
        h = self.dropout(h)

        h1 = self.relu(self.fc8(h))
        # print('h.size', h.size(),'h1.size()',h1.size())

        h_act = self.fc_act(h)
        # print('h_act',h_act.size())
        # print('h size after fc6', h.size())
        activity_logits = self.fc2_act(h_act)

        h1 = h1.unsqueeze(0)
        h1 = h1.transpose(0, 1)

        h_act = h_act.unsqueeze(0)
        h_act = h_act.transpose(0, 1)
        # print('h size after after', h.size())
        # x1_in = int(8192 / frames)
        # h = h.view(-1, frames, x1_in)

        # r_out, (h_n, h_c) = self.lstm1(x1)
        # r2 = r_out.contiguous()

        # print('Shape of r2: ', r2.shape)

        return h1, activity_logits, h_act

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
            # Conv1
            "features.0.weight": "conv1.weight",
            "features.0.bias": "conv1.bias",
            # Conv2
            "features.3.weight": "conv2.weight",
            "features.3.bias": "conv2.bias",
            # Conv3a
            "features.6.weight": "conv3a.weight",
            "features.6.bias": "conv3a.bias",
            # Conv3b
            "features.8.weight": "conv3b.weight",
            "features.8.bias": "conv3b.bias",
            # Conv4a
            "features.11.weight": "conv4a.weight",
            "features.11.bias": "conv4a.bias",
            # Conv4b
            "features.13.weight": "conv4b.weight",
            "features.13.bias": "conv4b.bias",
            # Conv5a
            "features.16.weight": "conv5a.weight",
            "features.16.bias": "conv5a.bias",
            # Conv5b
            "features.18.weight": "conv5b.weight",
            "features.18.bias": "conv5b.bias",
            # fc6
            "classifier.0.weight": "fc6.weight",
            "classifier.0.bias": "fc6.bias",
            # # fc7
            # "classifier.3.weight": "fc7.weight",
            # "classifier.3.bias": "fc7.bias",
        }

        p_dict = torch.load(Path.model_dir())
        # print('c3d p_dict {}'.format(p_dict))
        s_dict = self.state_dict()
        for name in p_dict:
            # print('p_dict name', name)
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
            # print('corresp_name[name]', corresp_name[name])
        self.load_state_dict(s_dict)


    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
         model.conv5a, model.conv5b, model.fc6]
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
    inputs = torch.rand(3, 3, 16, 112, 112)

    net = C3D_Encoder(pretrained=True)

    outputs = net.forward(inputs)

    # print(outputs)
    print('outputs:', outputs[0].size(), outputs[1].size(), outputs[2].size())

