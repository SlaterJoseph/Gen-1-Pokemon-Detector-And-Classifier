import timm
from torch.nn import Module


def build_model(model_name: str='resnet50', num_classes: int=152) -> Module:
    """
    Builds a model
    :param model_name: The model to train
    :param num_classes: The number of classes (152 for gen 1 + other)
    :return: The model
    """
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    return model