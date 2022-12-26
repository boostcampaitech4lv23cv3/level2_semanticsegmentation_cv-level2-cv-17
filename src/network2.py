import segmentation_models_pytorch as smp
import torch.optim as optim
import torch.nn as nn
import torch


    
def smp_criterion():
    return nn.CrossEntropyLoss()


def smp_model():
    model = smp.DeepLabV3Plus(
        encoder_name='efficientnet-b0',
        encoder_weights='imagenet',
        in_channels=3,
        classes=11
    )
    return model
    
def smp_optim(model,lr:float=0.0001):
    return optim.Adam(params = model.parameters(), lr = lr, weight_decay=1e-6)

    
    
    