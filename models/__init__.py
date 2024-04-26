# python3.7
"""Collects all models."""

from .pggan_generator import PGGANGenerator
from .pggan_discriminator import PGGANDiscriminator
from .stylegan_generator import StyleGANGenerator
from .stylegan_discriminator import StyleGANDiscriminator
from .stylegan2_generator import StyleGAN2Generator
from .stylegan2_discriminator import StyleGAN2Discriminator
from .stylegan3_generator import StyleGAN3Generator
from .ghfeat_encoder import GHFeatEncoder
from .perceptual_model import PerceptualModel
from .inception_model import InceptionModel
from .eg3d_generator import EG3DGenerator
from .eg3d_discriminator import DualDiscriminator
from .pigan_generator import PiGANGenerator
from .pigan_discriminator import PiGANDiscriminator
from .volumegan_generator import VolumeGANGenerator
from .volumegan_discriminator import VolumeGANDiscriminator
from .eg3d_generator_fv import EG3DGeneratorFV
from .bev3d_generator import BEV3DGenerator
from .berf3d_generator import BerfScene3DGenerator

__all__ = ['build_model']

_MODELS = {
    'PGGANGenerator': PGGANGenerator,
    'PGGANDiscriminator': PGGANDiscriminator,
    'StyleGANGenerator': StyleGANGenerator,
    'StyleGANDiscriminator': StyleGANDiscriminator,
    'StyleGAN2Generator': StyleGAN2Generator,
    'StyleGAN2Discriminator': StyleGAN2Discriminator,
    'StyleGAN3Generator': StyleGAN3Generator,
    'GHFeatEncoder': GHFeatEncoder,
    'PerceptualModel': PerceptualModel.build_model,
    'InceptionModel': InceptionModel.build_model,
    'EG3DGenerator': EG3DGenerator,
    'EG3DDiscriminator': DualDiscriminator,
    'PiGANGenerator': PiGANGenerator,
    'PiGANDiscriminator': PiGANDiscriminator,
    'VolumeGANGenerator': VolumeGANGenerator,
    'VolumeGANDiscriminator': VolumeGANDiscriminator,
    'EG3DGeneratorFV': EG3DGeneratorFV,
    'BEV3DGenerator': BEV3DGenerator,
    'BerfScene3DGenerator': BerfScene3DGenerator,
}


def build_model(model_type, **kwargs):
    """Builds a model based on its class type.

    Args:
        model_type: Class type to which the model belongs, which is case
            sensitive.
        **kwargs: Additional arguments to build the model.

    Raises:
        ValueError: If the `model_type` is not supported.
    """
    if model_type not in _MODELS:
        raise ValueError(f'Invalid model type: `{model_type}`!\n'
                         f'Types allowed: {list(_MODELS)}.')
    return _MODELS[model_type](**kwargs)
