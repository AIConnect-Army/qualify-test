import torch
from segmentation_models_pytorch.encoders._base import EncoderMixin
from models.swin.swin_transformer import SwinTransformer
from typing import List

class SwinEncoder(torch.nn.Module, EncoderMixin):

    def __init__(self, **kwargs):
        super().__init__()

        # A number of channels for each encoder feature tensor, list of integers
        self._out_channels: List[int] = [128, 256, 512, 1024]

        # A number of stages in decoder (in other words number of downsampling operations), integer
        # use in in forward pass to reduce number of returning features
        self._depth: int = 5

        # Default number of input channels in first Conv2d layer for encoder (usually 3)
        self._in_channels: int = 3
        kwargs.pop('depth')

        self.model = SwinTransformer(**kwargs)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outs = self.model(x)
        return list(outs)

    def load_state_dict(self, state_dict, **kwargs):
        self.model.load_state_dict(state_dict['model'], strict=False, **kwargs)