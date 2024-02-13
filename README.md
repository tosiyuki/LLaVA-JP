# LLaVA-JP
LLaVAの手法を使用して[llm-jp/llm-jp-1.3b-v1.0](https://huggingface.co/llm-jp/llm-jp-1.3b-v1.0)を学習さて画像に対応したマルチモーダルなLLMを学習させる。

LLaVA-JPの学習はRTX4090(24GB)一台で行われています。

[English document](docs/EN_README.md) is here.

# Release
- [2/13] [llava-jp-1.3b-v1.0-siglip-so400m-patch14-384](https://huggingface.co/toshi456/llava-jp-1.3b-v1.0-siglip-so400m-patch14-384)を公開: Image Encoderに[google/siglip-so400m-patch14-384](google/siglip-so400m-patch14-384)を使用してLLaVA-1.5の手法で学習させています

# 出力例
## 例1 入力：猫の隣には何がありますか？

![猫](imgs/sample1.jpg)

| モデル名| 出力 |
|:-----------|:------------|
|[llava-jp-1.3b-v1.0](https://huggingface.co/toshi456/llava-jp-1.3b-v1.0)| ノートパソコン|
|[llava-jp-1.3b-v1.0-siglip-so400m-patch14-384](https://huggingface.co/toshi456/llava-jp-1.3b-v1.0-siglip-so400m-patch14-384)| 猫の隣にはノートパソコンがある。|
|[turing-motors/heron-chat-blip-ja-stablelm-base-7b-v0](https://huggingface.co/turing-motors/heron-chat-blip-ja-stablelm-base-7b-v0)|猫はノートパソコンの上に座っている。|
|[stabilityai/japanese-stable-vlm](https://huggingface.co/stabilityai/japanese-stable-vlm)|ノートパソコン|


## 例2 入力：この画像の面白い点を教えてください？
![黄色い人](imgs/sample2.jpg)

| モデル名| 出力 |
|:-----------|:------------|
|[llava-jp-1.3b-v1.0](https://huggingface.co/toshi456/llava-jp-1.3b-v1.0)| 画像の中で、黄色いシャツを着た男性が、車の荷台に座って洗濯機を使っている。このシーンは、男性が日常生活の中で洗濯機を使っていることを示唆している。この男性は、おそらくは都市部で、おそらくは公共交通機関を利用して、洗濯機を使って服を洗濯しているのだろう。このシーンは、日常生活の中で洗濯機を使うことの重要性を強調している。|
|[llava-jp-1.3b-v1.0-siglip-so400m-patch14-384](https://huggingface.co/toshi456/llava-jp-1.3b-v1.0-siglip-so400m-patch14-384)| この画像の興味深い点は、黄色いタクシーの荷台に置かれたアイロン台の上に立つ男性である。アイロン台は通常、車の荷台に置かれるものではないため、これは珍しい光景である。この男性は、アイロン台を使って自分の服をアイロン掛けしているように見えるが、これは型破りでユーモラスな行為である。|
|[turing-motors/heron-chat-blip-ja-stablelm-base-7b-v0](https://huggingface.co/turing-motors/heron-chat-blip-ja-stablelm-base-7b-v0)|画像では、黄色いトラックの荷台で洗濯物を干している男性が描かれている。彼はトラックに寄りかかり、洗濯物を取り込んでいる。このシーンは、男性がトラックで仕事をしているか、トラックを運転していることを示唆している。トラックは、このシーンの中心的な焦点であり、男性の作業スペースを提供しているように見える。背景には、道路を走る複数の車が見え、おそらくこの地域で交通渋滞が発生している可能性がある。|
|[stabilityai/japanese-stable-vlm](https://huggingface.co/stabilityai/japanese-stable-vlm)|男は車の上で洗濯をしている|


# 学習手順
```
git clone https://github.com/tosiyuki/LLava-JP.git
```
## Stage1(事前学習)
```
bash scripts/pretrain/pretrain_llm_jp_1.3b_bf.sh
```

## Stage2(ファインチューニング)
```
bash scripts/finetune/finetune_llm_jp_1.3b_bf.sh
```

## Stage2(LoRAチューニング)
```
bash scripts/finetune/finetune_lora_llm_jp.sh
```

# 学習データ
## Stage1(事前学習)
- [STAIR Captions](https://github.com/STAIR-Lab-CIT/STAIR-captions)
- [LLaVA-CC3M-Pretrain-595K-JA](https://huggingface.co/datasets/toshi456/LLaVA-CC3M-Pretrain-595K-JA)

LLaVA-CC3M-Pretrain-595K-JAは[LLaVA-CC3M-Pretrain-595K](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K)を日本語訳したデータになります。翻訳には[cyberagent/calm2-7b-chat](https://qiita.com/cyberagent/calm2-7b-chat)を使用しています。

## Stage2(ファインチューニング)
- [Japanese Visual Genome VQA dataset](https://github.com/yahoojapan/ja-vg-vqa)
- [LLaVA-Instruct-150K-JA](https://huggingface.co/datasets/turing-motors/LLaVA-Instruct-150K-JA)

# 学習済みモデルの重み
## Pretrained
- [llava-pretrain-jp-1.3b-v1.0](https://huggingface.co/toshi456/llava-pretrain-jp-1.3b-v1.0)
## full training
- [llava-jp-1.3b-v1.0](https://huggingface.co/toshi456/llava-jp-1.3b-v1.0)
- [llava-jp-1.3b-v1.0-siglip-so400m-patch14-384](https://huggingface.co/toshi456/llava-jp-1.3b-v1.0-siglip-so400m-patch14-384)

# Acknowledgement
- [LLaVA](https://github.com/haotian-liu/LLaVA): LLaVA-JPを学習させるに当たりほとんどのコードがこの素晴らしいプロジェクトがベースとなっています。
- [llm-jp](https://github.com/llm-jp): llm-jpが大規模なモデルだけではなく1.3Bという小規模で高性能なベースモデルを開発しているおかげでLLaVA-JPの学習は成功しています

# TODO
- [x] LLaVA-CC3M-Pretrain-595K-JAの公開
- [x] llava-jpの重み公開(モデル名は変えるかもしれません)
- [ ] LoRAを使ったファインチューニングの実施