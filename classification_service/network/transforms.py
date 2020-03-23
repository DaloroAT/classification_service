from typing import List, Tuple, Union, Any

from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.core.composition import Compose as AlbCompose
import numpy as np
import cv2
import torchvision.transforms as transforms
from torch import Tensor

from classification_service.config import config


class Denormalize:

    def __init__(self):
        inverse_mean = [-x_mean / x_std for x_mean, x_std in zip(config.mean, config.std)]
        inverse_std = [1 / x_std for x_std in config.std]
        self.denormalization = transforms.Normalize(mean=inverse_mean, std=inverse_std)

    def __call__(self, image: Tensor) -> Tensor:
        return self.denormalization(image)


class LetterBox(ImageOnlyTransform):

    def __init__(self,
                 size_hw: Union[List[int], Tuple[int, int]],
                 fill_color: Union[List[int], Tuple[int, int, int], int] = 0):

        super(LetterBox, self).__init__(always_apply=True, p=1.0)

        assert size_hw[0] > 0 and size_hw[1] > 0 and len(size_hw) == 2

        self.size_hw = size_hw
        self.fill_color = fill_color

    def apply(self, image: np.ndarray, **params: Any):
        return letterbox_image(image, self.size_hw, self.fill_color).astype(np.uint8)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return "size_hw", "fill_color"


class To3Chan(ImageOnlyTransform):

    def __init__(self):

        super(To3Chan, self).__init__(always_apply=True, p=1.0)

    def apply(self, image: np.ndarray, **params: Any):
        return to_3_chan(image)

    def get_transform_init_args_names(self) -> Tuple:
        return ()


class BGR2RGB(ImageOnlyTransform):

    def __init__(self):

        super(BGR2RGB, self).__init__(always_apply=True, p=1.0)

    def apply(self, image: np.ndarray, **params: Any):
        return bgr2rgb(image)

    def get_transform_init_args_names(self) -> Tuple:
        return ()


class ComposeAlbumentationAndTorch:

    def __init__(self,
                 albumentation_compose: AlbCompose,
                 torch_compose: transforms.Compose):
        assert isinstance(albumentation_compose, AlbCompose)
        assert isinstance(torch_compose, transforms.Compose)

        self.albumentation_compose = albumentation_compose
        self.torch_compose = torch_compose

    def __call__(self, image: np.ndarray) -> Tensor:
        image_alb = self.albumentation_compose(image=image)["image"]
        image_torch = self.torch_compose(image_alb)
        return image_torch

    def __repr__(self) -> str:
        representation = f"{self.__class__.__name__}(" \
                         f"{self.albumentation_compose.__repr__()}, " \
                         f"{self.torch_compose.__repr__()})"
        return representation


class EvalTransforms:

    def __init__(self):
        self.eval_transform = ComposeAlbumentationAndTorch(get_base_alb_transforms(),
                                                           get_base_torch_transforms())

    def __call__(self, image: np.ndarray) -> Tensor:
        return self.eval_transform(image)

    def __repr__(self) -> str:
        return self.eval_transform.__repr__()


def letterbox_image(image: np.ndarray,
                    size_hw: Union[List[int], Tuple[int, int]],
                    fill_color: Union[List[int], Tuple[int, ...], int] = 0) -> np.ndarray:

    assert isinstance(image, np.ndarray)
    assert image.ndim == 3

    dst_h, dst_w = size_hw
    src_h, src_w = image.shape[:2]

    if src_w == dst_w and src_h == dst_h:
        return image

    if src_w / dst_w >= src_h / dst_h:
        scale = dst_w / src_w
    else:
        scale = dst_h / src_h

    if scale != 1.0:
        image_resized = cv2.resize(image,
                                   (int(scale * src_w), int(scale * src_h)),
                                   interpolation=cv2.INTER_LANCZOS4)
        resized_h, resized_w = image_resized.shape[:2]
    else:
        return image

    if src_w == resized_w and src_h == resized_h:
        return image_resized
    else:
        pad_w = (dst_w - resized_w) / 2
        pad_h = (dst_h - resized_h) / 2
        pad = (int(pad_w), int(pad_h), int(pad_w + 0.5), int(pad_h + 0.5))
        pad_width = ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0))

        image = np.pad(image_resized, pad_width, mode="constant", constant_values=fill_color)
        return image


def to_3_chan(image: np.ndarray) -> np.ndarray:
    assert isinstance(image, np.ndarray)

    h, w, *c = image.shape

    if len(c) == 1:
        c = c[0]
    elif len(c) == 0:
        c = 1
        image = image[:, :, np.newaxis]
    else:
        raise ValueError("Not supported shape")

    if c == 1:
        image = image.repeat(3, axis=2)
    elif c == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif c == 3:
        pass
    else:
        raise ValueError("Not supported number of channels")

    return image


def bgr2rgb(image: np.ndarray) -> np.ndarray:
    assert isinstance(image, np.ndarray)
    image_rgb = np.ascontiguousarray(image[:, :, ::-1])
    return image_rgb


def get_base_alb_transforms() -> AlbCompose:
    base_alb_transform = AlbCompose([To3Chan(),
                                     LetterBox(size_hw=(config.height, config.width), fill_color=config.fill_color),
                                     BGR2RGB()
                                     ])
    return base_alb_transform


def get_base_torch_transforms() -> transforms.Compose:
    base_torch_transform = transforms.Compose([transforms.ToTensor(),
                                               transforms.Normalize(mean=config.mean, std=config.std)])
    return base_torch_transform
