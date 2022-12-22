import segmentation_models_pytorch as smp
import torch.optim as optim
import torch.nn as nn
import torch


    
def smp_criterion():
    return smp.losses.FocalLoss(mode='multiclass')


def smp_model():
    model = smp.DeepLabV3(
        encoder_name='tu-eca_nfnet_l0',
        encoder_weights='imagenet',
        in_channels=3,
        classes=11
    )
    return model
    
def smp_optim(model,lr:float=1e-6):
    return optim.AdamW(params=model.parameters(), lr=lr)

    
    
    