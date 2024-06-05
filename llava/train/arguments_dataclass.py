from dataclasses import dataclass, field
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    base_model: Optional[str] = field(default="gpt2",
                                      metadata={"help": "gpt2 or gpt_neox or llama"})
    model_name_or_path: Optional[str] = field(default="rinna/japanese-gpt2-xsmall")
    version: Optional[str] = field(default="plain")
    freeze_backbone: bool = field(default=False) # LLMをFreezeするか
    tune_mm_mlp_adapter: bool = field(default=False) # 事前学習のときはmm_mlp_adapterだけ保存する.
    vision_tower: Optional[str] = field(default="openai/clip-vit-large-patch14-336")
    mm_vision_select_layer: Optional[int] = field(default=-2)   # default to the last two layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None) # fine-tuningのときには設定
    mm_projector_type: Optional[str] = field(default='mlp2x_gelu') # 2層の線形層
    mm_vision_select_feature: Optional[str] = field(default="patch")
    scales: Optional[list[float]] = field(default=None)
    vision_encoder_type: str = field(default="CLIP")
    tune_vision_tower: bool  = field(default=False)

    # ConvNeXt setting
    mm_vision_resolution: int = field(default=768)
    vision_add_five_stage: int = field(default=3)
    vision_five_stage_width: int = field(default=1536)
    drop_path_rates: Optional[list[float]] = field(default=None)
    tune_convnext_stage: Optional[list[str]] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(default="",
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default="/home/toshi/work/llava_jp/input/LLaVA-CC3M-Pretrain-595K/images",
                                        metadata={"help": "Path to image data."})
    image_aspect_ratio: str = 'square'
    image_size: Optional[int] = None


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=1024,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False) # dataset sampler option

    fp16: bool = field(default=False)
    bf16: bool = field(default=False)
    output_dir: str = field(default="./output_llava/checkpoints/llava-v1.5-japanese-gpt2-xsmall")
    num_train_epochs: int = field(default=1)
    per_device_train_batch_size: int = field(default=32)
    per_device_eval_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=1)
    evaluation_strategy: str = field(default="no")
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=24000)
    save_total_limit: int = field(default=1)
    learning_rate: float = field(default=1e-3)
    weight_decay: float = field(default=0.)
    warmup_ratio: float = field(default=0.03)
    logging_steps: int = field(default=1)
    model_max_length: int = field(default=1024)
    gradient_checkpointing: bool = field(default=True)
    dataloader_num_workers: int = field(default=16)
    lr_scheduler_type: str = field(default="cosine")
    seed: int = field(default=42)
