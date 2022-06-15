from pathlib import Path

import yaml

from ..embeddings import get_embeddings_extractor

def count_parameters(args):

    path = Path(__file__).resolve().parents[2]
    with open(path/("configs/data/"+args.config_data+".yaml"), 'r') as f:
        cfg_data = yaml.load(f, yaml.FullLoader)
    with open(path/("configs/classifier/"+args.config_class+".yaml"), 'r') as f:
        cfg_class = yaml.load(f, yaml.FullLoader)

    model = get_embeddings_extractor(
        n_mels=cfg_data['n_mels'],
        n_columns=cfg_data['n_columns'],
        **cfg_class['embeddings_extractor'],
    )
    return sum(p.numel() for p in model.parameters() if p.requires_grad)