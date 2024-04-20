# LLaVA-JP
LLaVAの手法を使用して[llm-jp/llm-jp-1.3b-v1.0](https://huggingface.co/llm-jp/llm-jp-1.3b-v1.0)のような軽量なLLMをベースに画像に対応したマルチモーダルなLVLMを学習させるためのコードです。

LLaVA-JPの学習はRTX4090(24GB)一台で行われています。

[English document](docs/EN_README.md) is here.

## Release
- [4/20] v1.1を公開: scaling_on_scalesを使用して768x768の高解像度画像を入力可能にしました。また、事前学習データをLLaVA-Pretrain-JA、ファインチューニングデータをLLaVA-v1.5-Instruct-620K-JAに変更しました。モデルは[llava-jp-1.3b-v1.1](https://huggingface.co/toshi456/llava-jp-1.3b-v1.1)で公開しています。
- [2/13] [llava-jp-1.3b-v1.0-siglip-so400m-patch14-384](https://huggingface.co/toshi456/llava-jp-1.3b-v1.0-siglip-so400m-patch14-384)を公開: Image Encoderに[google/siglip-so400m-patch14-384](google/siglip-so400m-patch14-384)を使用してLLaVA-1.5の手法で学習させています。

## Models
### Full Trained
|Model|Size|Train Data|Ver|
|-|-|-|-|
|[llava-jp-1.3b-v1.1](https://huggingface.co/toshi456/llava-jp-1.3b-v1.1)|1.86B|[LLaVA-Pretrain-JA](https://huggingface.co/datasets/turing-motors/LLaVA-Pretrain-JA), [LLaVA-v1.5-Instruct-620K-JA](https://huggingface.co/datasets/turing-motors/LLaVA-v1.5-Instruct-620K-JA)|v1.1|
|[llava-jp-1.3b-v1.0-620k](https://huggingface.co/toshi456/llava-jp-1.3b-v1.0)|1.86B|[LLaVA-Pretrain-JA](https://huggingface.co/datasets/turing-motors/LLaVA-Pretrain-JA), [LLaVA-v1.5-Instruct-620K-JA](https://huggingface.co/datasets/turing-motors/LLaVA-v1.5-Instruct-620K-JA)|v1.0|
|[llava-jp-1.3b-v1.0](https://huggingface.co/toshi456/llava-jp-1.3b-v1.0)|1.73B|[STAIR Captions](https://github.com/STAIR-Lab-CIT/STAIR-captions), [LLaVA-CC3M-Pretrain-595K-JA](https://huggingface.co/datasets/toshi456/LLaVA-CC3M-Pretrain-595K-JA), [Japanese Visual Genome VQA dataset](https://github.com/yahoojapan/ja-vg-vqa), [LLaVA-Instruct-150K-JA](https://huggingface.co/datasets/turing-motors/LLaVA-Instruct-150K-JA)|v1.0|
|[llava-jp-1.3b-v1.0-siglip-so400m-patch14-384](https://huggingface.co/toshi456/llava-jp-1.3b-v1.0-siglip-so400m-patch14-384)|1.86B|[STAIR Captions](https://github.com/STAIR-Lab-CIT/STAIR-captions), [LLaVA-CC3M-Pretrain-595K-JA](https://huggingface.co/datasets/toshi456/LLaVA-CC3M-Pretrain-595K-JA), [Japanese Visual Genome VQA dataset](https://github.com/yahoojapan/ja-vg-vqa), [LLaVA-Instruct-150K-JA](https://huggingface.co/datasets/turing-motors/LLaVA-Instruct-150K-JA)|v1.0|

### Pretrained
|Model|Train Data|Ver|
|-|-|-|
|[llava-jp-1.3b-v1.1-pretrain](https://huggingface.co/toshi456/llava-jp-1.3b-v1.1-pretrain)|[LLaVA-Pretrain-JA](https://huggingface.co/datasets/turing-motors/LLaVA-Pretrain-JA)|v1.1|
|[llava-pretrain-jp-1.3b-v1.0](https://huggingface.co/toshi456/llava-pretrain-jp-1.3b-v1.0)|[STAIR Captions](https://github.com/STAIR-Lab-CIT/STAIR-captions), [LLaVA-CC3M-Pretrain-595K-JA](https://huggingface.co/datasets/toshi456/LLaVA-CC3M-Pretrain-595K-JA)|v1.0|

### Comparing VLMs
|Model|JA-VG-VQA-500<br>(ROUGE-L)|JA-VLM-Bench-In-the-Wild<br>(ROUGE-L)|Heron-Bench(Detail)|Heron-Bench(Conv)|Heron-Bench(Complex)|Heron-Bench(Average)|
|-|-|-|-|-|-|-|
|[Japanese Stable VLM](https://huggingface.co/stabilityai/japanese-stable-vlm)|-|40.50|25.15|51.23|37.84|38.07|
|[EvoVLM-JP-v1-7B](https://huggingface.co/SakanaAI/EvoVLM-JP-v1-7B)|**19.70**|**51.25**|50.31|44.42|40.47|45.07|
|[Heron BLIP Japanese StableLM Base 7B llava-620k](https://huggingface.co/turing-motors/heron-chat-blip-ja-stablelm-base-7b-v1-llava-620k)|14.51|33.26|49.09|41.51|45.72|45.44|
|[Heron GIT Japanese StableLM Base 7B](https://huggingface.co/turing-motors/heron-chat-git-ja-stablelm-base-7b-v1)|15.18|37.82|42.77|**54.20**|43.53|46.83|
|[llava-jp-1.3b-v1.0-620k](https://huggingface.co/toshi456/llava-jp-1.3b-v1.0-620k)|12.69|44.58|**51.21**|41.05|45.95|44.84|
|[llava-jp-1.3b-v1.1](https://huggingface.co/toshi456/llava-jp-1.3b-v1.1)|13.33|44.40|50.00|51.83|**48.98**|**50.39**|

![評価結果](imgs/eval.png)

## 出力例
### 例1 猫の隣には何がありますか？

![猫](imgs/sample1.jpg)

| モデル名| 出力 |
|:-----------|:------------|
|[llava-jp-1.3b-v1.1](https://huggingface.co/toshi456/llava-jp-1.3b-v1.1)| 猫の隣にはノートパソコンがあります。|
|[llava-jp-1.3b-v1.0-620k](https://huggingface.co/toshi456/llava-jp-1.3b-v1.0-620k)| 猫の隣にはノートパソコンがあります。|
|[turing-motors/heron-chat-blip-ja-stablelm-base-7b-v0](https://huggingface.co/turing-motors/heron-chat-blip-ja-stablelm-base-7b-v0)|猫はノートパソコンの上に座っている。|
|[stabilityai/japanese-stable-vlm](https://huggingface.co/stabilityai/japanese-stable-vlm)|ノートパソコン|


### 例2 この画像の面白い点を教えてください？
![黄色い人](imgs/sample2.jpg)

| モデル名| 出力 |
|:-----------|:------------|
|[llava-jp-1.3b-v1.1](https://huggingface.co/toshi456/llava-jp-1.3b-v1.1)| この画像の面白い点は、黄色いタクシーの荷台に置かれた、服を脱いだままの洗濯機の上に立つ男性です。この男性は、おそらく洗濯物を乾燥させるために、この型破りな場所にいるようです。このシーンは、日常生活の中で型破りな場所に身を置くことの意外性と面白さを強調しています。|
|[llava-jp-1.3b-v1.0-620k](https://huggingface.co/toshi456/llava-jp-1.3b-v1.0-620k)| この画像の興味深い点は、黄色い車の荷台に置かれた洗濯機の上に立つ男性です。この珍しい配置は、洗濯機の上に立つ男性の姿を見ることができるため、興味深く、型破りなものとなっています。この男性は、洗濯機の上に立つことで、洗濯機の機能を妨げることなく、洗濯機の上に立っているように見えるのです。|
|[turing-motors/heron-chat-blip-ja-stablelm-base-7b-v0](https://huggingface.co/turing-motors/heron-chat-blip-ja-stablelm-base-7b-v0)|画像では、黄色いトラックの荷台で洗濯物を干している男性が描かれている。彼はトラックに寄りかかり、洗濯物を取り込んでいる。このシーンは、男性がトラックで仕事をしているか、トラックを運転していることを示唆している。トラックは、このシーンの中心的な焦点であり、男性の作業スペースを提供しているように見える。背景には、道路を走る複数の車が見え、おそらくこの地域で交通渋滞が発生している可能性がある。|
|[stabilityai/japanese-stable-vlm](https://huggingface.co/stabilityai/japanese-stable-vlm)|男は車の上で洗濯をしている|


## 学習手順
```
git clone https://github.com/tosiyuki/LLava-JP.git
```
### Stage1(事前学習)
```
bash scripts/pretrain/pretrain_llm_jp_1.3b_v1.1.sh
```

### Stage2(ファインチューニング)
```
bash scripts/finetune/finetune_llm-jp-1.3b-v1.1.sh
```

### Stage2(LoRAチューニング)
```
bash scripts/finetune/finetune_lora_llm_jp.sh
```

## Acknowledgement
- [LLaVA](https://github.com/haotian-liu/LLaVA): LLaVA-JPを学習させるに当たりほとんどのコードがこの素晴らしいプロジェクトがベースとなっています。
- [llm-jp](https://github.com/llm-jp): llm-jpが大規模なモデルだけではなく1.3Bという小規模で高性能なベースモデルを開発しているおかげでLLaVA-JPの学習は成功しています
- [scaling_on_scales](https://github.com/bfshi/scaling_on_scales/tree/master): 高解像度画像入力の対応はscaling_on_scalesの簡潔かつ分かりやすいコードのおかげで行えています。
