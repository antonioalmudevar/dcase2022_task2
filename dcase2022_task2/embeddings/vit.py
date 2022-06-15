import torch
from torch import nn

from einops import repeat

from ..blocks import Transformer

class ViT(nn.Module):
    def __init__(
        self, 
        n_mels: int,
        n_columns: int,
        embedding_dim: int,
        depth: int, 
        heads: int, 
        mlp_dim: int, 
        pool = 'cls',
        dropout = 0., 
        emb_dropout = 0.
    ):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Linear(n_mels, embedding_dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, n_columns + 1, embedding_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(embedding_dim, depth, heads, mlp_dim, dropout)

        self.pool = pool

        self.segment6 = nn.Linear(2*embedding_dim, embedding_dim)
        self.segment7 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, spec):
        spec = spec[:,0].permute(0,2,1)
        x = self.to_patch_embedding(spec)
        b, n, _ = x.shape
        
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        ### Stat Pool
        '''
        mean, std = torch.mean(x,1), torch.std(x,1)
        x = torch.cat((mean,std),1)
        x = self.segment6(x)
        x = self.segment7(x)
        '''

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        return x