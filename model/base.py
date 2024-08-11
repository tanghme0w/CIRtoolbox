import torch
import torch.nn as nn

class DualEncoderModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.di = nn.Parameter(torch.empty(0))  # device indicator

    def query_forward(self, img, text):
        raise NotImplementedError("query forward not implemented")

    def target_forward(self, img):
        raise NotImplementedError("target forward not implemented")
