import requests
import torch
import transformers
from PIL import Image

from transformers.generation.streamers import TextStreamer
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.llava_gpt2 import LlavaGpt2ForCausalLM
from llava.train.arguments_dataclass import ModelArguments, DataArguments, TrainingArguments
from llava.train.dataset import tokenizer_image_token


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_path = 'toshi456/llava-jp-1.3b-v1.0'
    model_args.vision_tower = "openai/clip-vit-large-patch14-336"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device=="cuda" else torch.float32

    model = LlavaGpt2ForCausalLM.from_pretrained(
        model_path, 
        low_cpu_mem_usage=True,
        use_safetensors=True,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=1024,
        padding_side="right",
        use_fast=False,
    )
    model.eval()

    conv_mode = "v1"
    conv = conv_templates[conv_mode].copy()

    # image pre-process
    image_url = "https://huggingface.co/rinna/bilingual-gpt-neox-4b-minigpt4/resolve/main/sample.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
    if device == "cuda":
        image_tensor = model.get_model().vision_tower.image_processor(image, return_tensors='pt')['pixel_values'].half().cuda().to(torch_dtype)
    else:
        image_tensor = model.get_model().vision_tower.image_processor(image, return_tensors='pt')['pixel_values'].to(torch_dtype)

    # create prompt
    # ユーザー: <image>\n{prompt}
    prompt = "猫の隣には何がありますか？"
    inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt, 
        tokenizer, 
        IMAGE_TOKEN_INDEX, 
        return_tensors='pt'
    ).unsqueeze(0)
    if device == "cuda":
        input_ids = input_ids.to(device)

    input_ids = input_ids[:, :-1] # </sep>がinputの最後に入るので削除する
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    streamer = TextStreamer(tokenizer, skip_prompt=True, timeout=20.0)

    # predict
    with torch.inference_mode():
        model.generate(
            inputs=input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.01,
            top_p=1.0,
            max_new_tokens=256,
            streamer=streamer,
            use_cache=True,
        )
