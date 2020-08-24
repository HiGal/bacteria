from albumentations import (
    PadIfNeeded,
    Resize,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    GaussNoise,
    RandomGamma
)
from albumentations.pytorch import ToTensorV2
import albumentations as A


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing():
    _transform = [
        A.Lambda(image=preprocessing_fn),
        Resize(p=1, height=512, width=512),
        A.Lambda(image=to_tensor, mask=to_tensor)
    ]
    return A.Compose(_transform)


def get_train_transforms(preprocessing_fn=None):
    result = Compose([
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        OneOf([
            ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            GridDistortion(p=0.5),
            OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
        ], p=0.8),
        # CLAHE(p=0.8),
        GaussNoise(),
        RandomBrightnessContrast(p=0.8),
        # RandomGamma(p=0.5),
        Resize(p=1, height=256, width=320),
    ], p=1)
    return result


def get_test_transforms(preprocessing_fn=None):
    result = Compose([
        Resize(p=1, height=256, width=320),
    ], p=1)
    return result
