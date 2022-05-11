import torch
import torch.nn as nn

randomseed = 0
torch.manual_seed(randomseed)
torch.cuda.manual_seed_all(randomseed)

class LSTM_ANNO(nn.Module):
    def __init__(self, num_classes=5):
        super(LSTM_ANNO, self).__init__()
        # defining encoder LSTM layers
        feat_dim = 512
        self.clip_len = 16
        self.lstm_hidden_size = 512
        self.num_classes = num_classes
        self.gru1 = nn.GRU(feat_dim, feat_dim//2, 2, batch_first=True, bidirectional=True)   # Fusion net: 1152, Concate: 1024, C3D: 4096, RES3D: 512
        self.gru2 = nn.GRU(feat_dim, feat_dim//2, num_layers=2, batch_first=True, bidirectional=True)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(feat_dim, feat_dim//4)
        # self.bn1 = nn.BatchNorm1d(512, momentum=0.01)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_final_score = nn.Linear(feat_dim//4, self.num_classes)

    def forward(self, x):
        state = None
        gru_output, (h_n) = self.gru1(x, state)
        gru1_output = gru_output.clone()
        gru_output, (h_n) = self.gru2(gru_output)
        gru_output = self.relu(self.fc1(gru_output[:, -1, :]))
        final_score = self.fc_final_score(gru_output)

        return final_score, gru1_output

if __name__ == "__main__":
    inputs = torch.rand(3, 8, 512)
    net = LSTM_ANNO(num_classes=5)
    outputs = net.forward(inputs)

    print(outputs[0].size(), outputs[1].size())
