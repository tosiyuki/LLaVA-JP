import open_clip
import timm
import torch
import torch.nn as nn
from open_clip.factory import HF_HUB_PREFIX, _get_hf_config
from open_clip.transform import image_transform_v2, PreprocessCfg
#from timm.models.convnext import ConvNeXtStage
from transformers import CLIPImageProcessor
from transformers import ConvNextModel, ConvNextConfig
from timm.layers import LayerNorm2d, LayerNorm

from llava.model.image_models.convnext import ConvNeXtStage, convnext_base, convnext_large


class ConvNeXtCLIPVisionTower(nn.Module):
    def __init__(
        self,
        args,
        vision_tower_name: str="laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft", 
        delay_load: bool=False
    ):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower_name
        self.update_resolution = getattr(
            args, 'mm_vision_resolution', 768
        )
        self.vision_add_five_stage = getattr(args, 'vision_add_five_stage', 0)
        self.vision_five_stage_width = getattr(args, 'vision_five_stage_width', 1536)
        self.drop_path_rates = getattr(args, 'drop_path_rates', None)

        self._load_convnext = {
            "convnext_base": convnext_base,
            "convnext_large": convnext_large,
        }

        if not delay_load:
            self.load_model()
        else:
            print(f"deloy_load vision tower is: {self.vision_tower_name}")
            if self.vision_tower_name.startswith(HF_HUB_PREFIX):
                self.cfg_only = _get_hf_config(self.vision_tower_name[len(HF_HUB_PREFIX):])["model_cfg"]["vision_cfg"]
            else:
                self.cfg_only = {
                    'image_size': self.update_resolution
                }

    def load_model(self):
        print(f"entering load model, load {self.vision_tower_name}")
        
        if self.vision_tower_name.startswith(HF_HUB_PREFIX):
            # open_clipのモデル
            model, _ = open_clip.create_model_from_pretrained(self.vision_tower_name)
            model.visual.preprocess_cfg["size"] = (self.update_resolution, self.update_resolution)
            pp_cfg = PreprocessCfg(**model.visual.preprocess_cfg)
            self.vision_tower = model.visual.trunk
            self.vision_tower_conf = _get_hf_config(self.vision_tower_name[len(HF_HUB_PREFIX):])["model_cfg"]["vision_cfg"]
        elif self.vision_tower_name in self._load_convnext.keys():
            pp_cfg = PreprocessCfg(
                size=(self.update_resolution, self.update_resolution), 
                mode='RGB', 
                mean=[0.48145466, 0.4578275, 0.40821073], 
                std=[0.26862954, 0.26130258, 0.27577711], 
                interpolation='bicubic', 
                resize_mode='shortest', 
                fill_color=0
            )
            self.vision_tower = self._load_convnext[self.vision_tower_name]()
            self.vision_tower_conf = {
                'image_size': self.update_resolution
            }
        else:
            raise ValueError(f"self.vision_tower_name: {self.vision_tower_name} not found!")

        self.image_processor = image_transform_v2(pp_cfg, False)
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

        if self.update_resolution > 256:
            self.set_crop_size(self.update_resolution)
            print(
                f'Crop size changed to {self.update_resolution}x{self.update_resolution}')

        if self.vision_add_five_stage != 0:
            self.add_stage(self.vision_add_five_stage, self.vision_five_stage_width)
            print(
                f'Added stage with width {self.vision_five_stage_width}')
                
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                # Get the image features
                image_feature = self.vision_tower(image)
                image_feature = image_feature.permute(0, 2, 3, 1)
                image_feature = image_feature.reshape(image_feature.shape[0], -1, image_feature.shape[3]).to(images.dtype)

                image_features.append(image_feature)
        else:
            # Get the image features
            image_features = self.vision_tower(images)
            image_features = image_features.permute(0, 2, 3, 1)
            image_features = image_features.reshape(image_features.shape[0], -1, image_features.shape[3]).to(images.dtype)

        return image_features

    def set_crop_size(self, new_size):
        size_dict = {'height': new_size, 'width': new_size}
        self.image_processor.crop_size = size_dict
        self.image_processor.size = {"shortest_edge": new_size}
        self.vision_tower_conf['image_size'] = new_size

    def add_stage(self, depths=3, hidden_dims=3072):
        self.vision_tower.stages.add_module(
            '4', 
            ConvNeXtStage(
                1536, #self.hidden_size, 
                hidden_dims,
                depth=depths,
                drop_path_rates=self.drop_path_rates,
                norm_layer=LayerNorm2d,
                norm_layer_cl=LayerNorm,
            )
        )
        self.vision_tower.head = nn.Identity()

    def save_config(self, path):
        self.vision_tower_conf.save_pretrained(path)

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
            return self.vision_tower_conf
        
        #return self.cfg_only
        return self.cfg_only

    @property
    def hidden_size(self):
        return self.vision_five_stage_width

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config['image_size'] // 32) ** 2

    @property
    def crop_size(self):
        return self.image_processor.crop_size