import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math

class ArcFace(nn.Module):

    def __init__(
        self, 
        embedding_dim:int, 
        n_classes: int,
        norm: str = None,
        s: float = 64.0, 
        m: float = 0.50, 
        easy_margin: bool = False
    ) -> None:
        super().__init__()
        if norm is None:
            self.layer_norm = nn.Identity(embedding_dim)
        elif norm.upper()=="BATCH":
            self.layer_norm = nn.BatchNorm1d(embedding_dim)
        elif norm.upper()=="LAYER":
            self.layer_norm = nn.LayerNorm(embedding_dim)

        self.embedding_dim = embedding_dim
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(n_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, labels):
        input = self.layer_norm(input)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        output = (labels * phi) + ((1.0 - labels) * cosine) 
        output *= self.s
        return output