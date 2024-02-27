import math
import re

import torch
import torch.nn as nn


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}
    

class FeatureIRLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

class TokenDownLayer(nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        self.dwn = nn.Sequential(
            nn.AdaptiveAvgPool2d(shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        if h * h == num_tokens:
            x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        else:
            # FIXME サイズによっては失敗する
            w = int(num_tokens/h)
            assert w*h == num_tokens
            x = x.permute(0, 2, 1).reshape(b, -1, w, h)

        x = self.dwn(x)
        x = x.flatten(2).transpose(1, 2)
        return x
    

class PosInjectLayer(nn.Module):
    # https://github.com/Meituan-AutoML/Twins/blob/main/gvt.py
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1) -> None:
        super().__init__()
        self.peg = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride, 1, bias=True, groups=out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens
        cnn_feat = x.transpose(1, 2).view(b, c, h, h)
        x = self.peg(cnn_feat) + cnn_feat
        x = x.flatten(2).transpose(1, 2)
        return x


class LDPNetV2Projector(nn.Module):
    # https://github.com/Meituan-AutoML/MobileVLM/blob/main/mobilevlm/model/vision_projector.py
    def __init__(self, config=None):
        super().__init__()
        inc, ouc = config.mm_hidden_size, config.hidden_size
        self.mlp = FeatureIRLayer(inc, ouc)
        self.dwn = TokenDownLayer((12, 12))
        self.peg = PosInjectLayer(ouc, ouc, stride=1)

    def forward(self, x):
        x = self.mlp(x)
        x = self.dwn(x)
        x = self.peg(x)
        return x
    

def get_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    elif projector_type == 'identity':
        return IdentityMap()
    elif projector_type == 'ldpnetv2':
        return LDPNetV2Projector(config)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    raise ValueError(f'Unknown projector type: {projector_type}')