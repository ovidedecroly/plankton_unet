import segmentation_models_pytorch as smp

def get_unet_model(name='efficientnet-b3', in_channels=3, classes=1, activation=None):
    model = smp.Unet(encoder_name=name, in_channels=in_channels, classes=classes, activation=activation)
    return model
