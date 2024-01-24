import torch.nn as nn

class CalisthenicsNet(nn.Module):
    def __init__(self):
        super(CalisthenicsNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 224 * 224, 1)  

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x
