from typing import Optional

import torch
import torch.nn as nn

from transformers import (
    CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig,\
    SiglipVisionModel, SiglipImageProcessor, SiglipVisionConfig
)
from llava.s2wrapper import forward as multiscale_forward


class CLIPVisionTower(nn.Module):
    def __init__(
        self, 
        vision_tower_name: str="openai/clip-vit-large-patch14-336", 
        mm_vision_select_layer: int=-2, # v1.5 is -2
        mm_vision_select_feature: str="patch",
        delay_load: bool=False,
        requires_grad: bool=False,
        scales: Optional[float] = None
    ):
        super().__init__()

        self.is_loaded = False
        self.requires_grad = requires_grad
        self.scales = scales

        self.vision_tower_name = vision_tower_name
        self.select_layer = mm_vision_select_layer
        self.select_feature = mm_vision_select_feature

        self.image_processor = None
        self.vision_tower = None

        if not delay_load:
            self.load_model()
        else:
            if "clip" in self.vision_tower_name:
                self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
            elif "siglip" in self.vision_tower_name:
                self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name)
            else:
                raise ValueError(f'Unsupported vision_tower_name: {self.vision_tower_name}')

    def load_model(self):
        if "clip" in self.vision_tower_name:
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        elif "siglip" in self.vision_tower_name:
            self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
            self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name)
        else:
            raise ValueError(f'Unsupported vision_tower_name: {self.vision_tower_name}')
        self.vision_tower.requires_grad_(self.requires_grad)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                if self.scales is None:
                    image_feature = self._forward_feature(images.unsqueeze(0))
                else:
                    image_feature = multiscale_forward(
                        self._forward_feature, 
                        images.unsqueeze(0), 
                        scales=self.scales, 
                        num_prefix_token=0, 
                        max_split_size=self.image_processor.size["height"]
                    )
                #image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            if self.scales is None:
                image_features = self._forward_feature(images)
            else:
                image_features = multiscale_forward(
                    self._forward_feature, 
                    images, 
                    scales=self.scales, 
                    num_prefix_token=0, 
                    max_split_size=self.image_processor.size["height"]
                )
            #image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features
    
    def _forward_feature(self, inputs):
        return self.feature_select(self.vision_tower(inputs.to(device=self.device, dtype=self.dtype), output_hidden_states=True))

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        if self.scales is None:
            return self.config.hidden_size
        
        return self.config.hidden_size*len(self.scales)

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
