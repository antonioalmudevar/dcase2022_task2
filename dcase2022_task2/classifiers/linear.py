from torch import nn


class Linear(nn.Module):
    def __init__(
        self, 
        embedding_dim: int,
        n_classes: int,
        norm: str = None
    ) -> None:
        super().__init__()

        if norm is None:
            layer_norm = nn.Identity
        elif norm.upper()=="BATCH":
            layer_norm = nn.BatchNorm1d
        elif norm.upper()=="LAYER":
            layer_norm = nn.LayerNorm
            
        self.net = nn.Sequential(
            layer_norm(embedding_dim),
            nn.Linear(embedding_dim, n_classes)
        )

    def forward(self, embed, labels=None):
        x = self.net(embed)
        return x