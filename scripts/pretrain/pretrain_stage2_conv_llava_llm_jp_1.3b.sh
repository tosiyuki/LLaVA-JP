#!/bin/bash

python train_llava.py \
    --model_name_or_path ./output_llava/checkpoints/pretrain-conv-llava-jp-1.3b-stage1-768 \
    --version plain \
    --freeze_backbone False \
    --tune_mm_mlp_adapter False \
    --vision_tower convnext_large \
    --mm_projector_type mlp2x_gelu \
    --tune_vision_tower True \
    --vision_encoder_type ConvNeXt \
    --mm_vision_resolution 768 \
    --vision_add_five_stage 6 \
    --vision_five_stage_width 3072 \
    --drop_path_rates 0.085 0.088 0.091 0.094 0.097 0.100 \
    --data_path ./dataset/llava_pretrain_blip_laion_cc_sbu_558k_ja.json \
    --lazy_preprocess False \
    --is_multimodal True \
    --image_folder ~/datasets/images \
    --image_aspect_ratio square \
    --image_size 768 \
    --optim adamw_bnb_8bit \
    --double_quant True \
    --quant_type nf4 \
    --bits 16 \
    --lora_enable False \
    --group_by_modality_length False \
    --fp16 False \
    --bf16 True \
    --output_dir ./output_llava/checkpoints/pretrain-conv-llava-jp-1.3b-stage2-768 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --model_max_length 1532 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lr_scheduler_type "cosine"
