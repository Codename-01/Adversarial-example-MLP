import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1 = nn.Linear(784,100)
        self.L3 = nn.Linear(100,10)

    def forward(self, xb):
        xb = xb.view(-1, 28*28)
        Z1 = F.relu(self.L1(xb))
        return F.softmax(self.L3(Z1))