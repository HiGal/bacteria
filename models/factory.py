import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from segmentation_models_pytorch.base import SegmentationHead


def get_net(model_name='resnet34'):
    aux_params = dict(
        pooling='avg',  # one of 'avg', 'max'
        dropout=0.5,  # dropout ratio, default is None
        activation='sigmoid',  # activation function, default is None
        classes=6,  # define number of output labels
    )
    model = smp.FPN(model_name, encoder_weights='imagenet', activation='sigmoid', aux_params=aux_params)
    # model.segmentation_head = SegmentationHead(
    #     in_channels=model.decoder.out_channels,
    #     out_channels=1,
    #     kernel_size=1
    # )
    preprocess_params = get_preprocessing_fn(model_name, pretrained='imagenet')
    return model, preprocess_params
