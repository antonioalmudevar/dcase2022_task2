from torch import nn

from ..blocks import LinearBlock

class Net(nn.Module):
    def __init__(
        self, 
        embedding_dim: int,
        n_classes: int,
        norm: str = None,
        dropout: int = 0,
    ):
        super().__init__()

        if norm is None:
            layer_norm = nn.Identity
        elif norm.upper()=="BATCH":
            layer_norm = nn.BatchNorm1d
        elif norm.upper()=="LAYER":
            layer_norm = nn.LayerNorm

        self.net = nn.Sequential(
            layer_norm(embedding_dim),
            nn.ReLU(inplace=True),
            LinearBlock(embedding_dim, embedding_dim//4, norm=norm, dropout=dropout),
            #LinearBlock(embedding_dim//2, embedding_dim//4, norm=norm),
            nn.Linear(embedding_dim//4, n_classes),
        )

    def forward(self, embed):
        x = self.net(embed)
        return x