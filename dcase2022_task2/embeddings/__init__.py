def get_embeddings_extractor(**kwargs):
    arch = kwargs['arch']
    kwargs.pop('arch', None)
    if arch.upper() == 'VIT':
        from .vit import ViT
        return ViT(**kwargs)
    elif arch.upper() == 'MOBILENETV2':
        kwargs.pop('n_mels', None)
        kwargs.pop('n_columns', None)
        from .mobilenetv2 import mobilenetv2
        return mobilenetv2(**kwargs)
    elif arch.upper() == 'X-VECTOR':
        kwargs.pop('n_columns', None)
        from .xvector import XVector
        return XVector(**kwargs)