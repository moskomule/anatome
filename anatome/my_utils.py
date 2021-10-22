import torch.nn as nn

def remove_hook(mdl: nn.Module, hook):
    """
    ref: https://github.com/pytorch/pytorch/issues/5037
    """
    handle = mdl.register_forward_hook(hook)
    handle.remove()


def hello():
    print('hello using my-anatome with library name anatome')

def my_anatome_test():
    import torch
    from torchvision.models import resnet18
    from anatome import DistanceHook
    from anatome.my_utils import remove_hook

    model = resnet18()
    hook1 = DistanceHook(model, "layer3.0.conv1")
    hook2 = DistanceHook(model, "layer3.0.conv2")
    model.eval()
    with torch.no_grad():
        model(torch.randn(128, 3, 224, 224))
    # downsampling to (size, size) may be helpful
    hook1.distance(hook2, size=8)
    hook1.clear()
    hook2.clear()
    remove_hook(model, hook1)
    remove_hook(model, hook2)


if __name__ == '__main__':
    my_anatome_test()
    print('Done, success!\a\n')