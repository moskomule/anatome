def remove_hook(mdl: nn.Module, hook):
    """
    ref: https://github.com/pytorch/pytorch/issues/5037
    """
    handle = mdl.register_forward_hook(hook)
    handle.remove()
