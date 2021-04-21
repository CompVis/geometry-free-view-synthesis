import torch
import torch.nn as nn
import torch.nn.functional as F

class DummyLoss(nn.Module):
    def __init__(self):
        super().__init__()
