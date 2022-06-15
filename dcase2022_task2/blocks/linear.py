from torch import nn
from torch.nn.modules.dropout import Dropout

class LinearBlock(nn.Module):

    def __init__(
        self, 
        dim_in: int, 
        dim_out: int,
        dropout: float = 0,
        act: str = "relu",
        norm: str = "batch",
    ):
        super().__init__()

        if act is None:
            layer_act = nn.Identity
        elif act.upper()=="RELU":
            layer_act = nn.ReLU

        if norm is None:
            layer_norm = nn.Identity
        elif norm.upper()=="BATCH":
            layer_norm = nn.BatchNorm1d
        elif norm.upper()=="LAYER":
            layer_norm = nn.LayerNorm

        self.layer = nn.Sequential(
            nn.Linear(dim_in, dim_out),
            layer_norm(dim_out),
            layer_act(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.layer(x)
        return x