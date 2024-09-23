from efficientnet_pytorch import EfficientNet
from densenet_pytorch import DenseNet 
import torchvision
import torch.nn as nn



def init_efficientnet(model_name, num_classes):
    return EfficientNet.from_pretrained(model_name, num_classes=num_classes)
    # if model_type == ModelType.regr:
        # num_ftrs = model._fc.in_features
        # model._fc = nn.Linear(num_ftrs, 1)

def init_densenet(model_name, num_classes):
    model = DenseNet.from_pretrained(model_name)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
    return model


def init_resnet(model_name, num_classes):
    model = torchvision.models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model



def initialize_model(model_name, device, num_classes):
    if "efficientnet" in model_name:
        model = init_efficientnet(model_name, num_classes)

    elif 'densenet' in model_name:
        model = init_densenet(model_name, num_classes)

    elif'resnet' in model_name:
        model = init_resnet(model_name, num_classes)

    else:
        print('ERROR: Unabble initialize model because "' + model_name + '" is incorrect model name.')
        return None
    
    model.to(device)
    return model