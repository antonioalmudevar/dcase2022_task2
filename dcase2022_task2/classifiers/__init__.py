def get_classifier(**kwargs):
    arch = kwargs['arch']
    kwargs.pop('arch', None)
    if arch.upper() == 'LINEAR':
        from .linear import Linear
        return Linear(**kwargs)
    elif arch.upper() == 'ARCFACE':
        from .arcface import ArcFace
        return ArcFace(**kwargs)
    elif arch.upper() == 'NET':
        from .net import Net
        return Net(**kwargs)