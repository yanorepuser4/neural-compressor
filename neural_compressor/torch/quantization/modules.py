import torch
import torch.nn as nn



class Matmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)

class BatchMatmul(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.bmm(x,y)