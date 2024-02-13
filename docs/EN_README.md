# LLaVA-JP
Use the LLaVA method to train [llm-jp/llm-jp-1.3b-v1.0](https://huggingface.co/llm-jp/llm-jp-1.3b-v1.0) to learn VLM.

LLaVA-JP is learned with a single RTX4090(24GB).

[Japanese](../README.md)

# Release
- [2/13] [llava-jp-1.3b-v1.0-siglip-so400m-patch14-384](https://huggingface.co/toshi456/llava-jp-1.3b-v1.0-siglip-so400m-patch14-384)is released: [google/siglip-so400m-patch14-384](google/siglip-so400m-patch14-384) is used for Image Encoder and trained with LLaVA-1.5 method

# Output example
## Ex.1 Input: 猫の隣には何がありますか？

![猫](../imgs/sample1.jpg)

| モデル名| 出力 |
|:-----------|:------------|
|[llava-jp-1.3b-v1.0](https://huggingface.co/toshi456/llava-jp-1.3b-v1.0)| ノートパソコン|
|[llava-jp-1.3b-v1.0-siglip-so400m-patch14-384](https://huggingface.co/toshi456/llava-jp-1.3b-v1.0-siglip-so400m-patch14-384)| 猫の隣にはノートパソコンがある。|
|[turing-motors/heron-chat-blip-ja-stablelm-base-7b-v0](https://huggingface.co/turing-motors/heron-chat-blip-ja-stablelm-base-7b-v0)|猫はノートパソコンの上に座っている。|
|[stabilityai/japanese-stable-vlm](https://huggingface.co/stabilityai/japanese-stable-vlm)|ノートパソコン|


## Ex.2 Input: この画像の面白い点を教えてください？
![黄色い人](../imgs/sample2.jpg)

| モデル名| 出力 |
|:-----------|:------------|
|[llava-jp-1.3b-v1.0](https://huggingface.co/toshi456/llava-jp-1.3b-v1.0)| 画像の中で、黄色いシャツを着た男性が、車の荷台に座って洗濯機を使っている。このシーンは、男性が日常生活の中で洗濯機を使っていることを示唆している。この男性は、おそらくは都市部で、おそらくは公共交通機関を利用して、洗濯機を使って服を洗濯しているのだろう。このシーンは、日常生活の中で洗濯機を使うことの重要性を強調している。|
|[llava-jp-1.3b-v1.0-siglip-so400m-patch14-384](https://huggingface.co/toshi456/llava-jp-1.3b-v1.0-siglip-so400m-patch14-384)| この画像の興味深い点は、黄色いタクシーの荷台に置かれたアイロン台の上に立つ男性である。アイロン台は通常、車の荷台に置かれるものではないため、これは珍しい光景である。この男性は、アイロン台を使って自分の服をアイロン掛けしているように見えるが、これは型破りでユーモラスな行為である。|
|[turing-motors/heron-chat-blip-ja-stablelm-base-7b-v0](https://huggingface.co/turing-motors/heron-chat-blip-ja-stablelm-base-7b-v0)|画像では、黄色いトラックの荷台で洗濯物を干している男性が描かれている。彼はトラックに寄りかかり、洗濯物を取り込んでいる。このシーンは、男性がトラックで仕事をしているか、トラックを運転していることを示唆している。トラックは、このシーンの中心的な焦点であり、男性の作業スペースを提供しているように見える。背景には、道路を走る複数の車が見え、おそらくこの地域で交通渋滞が発生している可能性がある。|
|[stabilityai/japanese-stable-vlm](https://huggingface.co/stabilityai/japanese-stable-vlm)|男は車の上で洗濯をしている|


# Train
```
git clone https://github.com/tosiyuki/LLava-JP.git
```
## Stage1(Pretrain)
```
bash scripts/pretrain/pretrain_llm_jp_1.3b_bf.sh
```

## Stage2(Fine-tuning)
```
bash scripts/finetune/finetune_llm_jp_1.3b_bf.sh
```

## Stage2(Fine-tuning by LoRA)
```
bash scripts/finetune/finetune_lora_llm_jp.sh
```

# Training data
## Stage1(Pretrain)
- [STAIR Captions](https://github.com/STAIR-Lab-CIT/STAIR-captions)
- [LLaVA-CC3M-Pretrain-595K-JA](https://huggingface.co/datasets/toshi456/LLaVA-CC3M-Pretrain-595K-JA)

LLaVA-CC3M-Pretrain-595K-JA is a Japanese translation of [LLaVA-CC3M-Pretrain-595K](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K). The translation was done using[cyberagent/calm2-7b-chat](https://qiita.com/cyberagent/calm2-7b-chat).

## Stage2(Fine-tuning)
- [Japanese Visual Genome VQA dataset](https://github.com/yahoojapan/ja-vg-vqa)
- [LLaVA-Instruct-150K-JA](https://huggingface.co/datasets/turing-motors/LLaVA-Instruct-150K-JA)

# About releasing weights
## Pretrained
- [llava-pretrain-jp-1.3b-v1.0](https://huggingface.co/toshi456/llava-pretrain-jp-1.3b-v1.0)
## full training
- [llava-jp-1.3b-v1.0](https://huggingface.co/toshi456/llava-jp-1.3b-v1.0)
- [llava-jp-1.3b-v1.0-siglip-so400m-patch14-384](https://huggingface.co/toshi456/llava-jp-1.3b-v1.0-siglip-so400m-patch14-384)

# Acknowledgement
- [LLaVA](https://github.com/haotian-liu/LLaVA): Most of the code used to train LLaVA-JP is based on this excellent project.
- [llm-jp](https://github.com/llm-jp): LLaVA-JP's learning has been successful thanks to the fact that llm-jp has developed not only a large model, but also a small, high-performance base model of 1.3B