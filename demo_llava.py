# TODO 重みをアップロードするタイミングで修正する
import torch
import transformers
from PIL import Image

from transformers.generation.streamers import TextIteratorStreamer
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.llava_gpt2 import LlavaGpt2ForCausalLM
from llava.model.llava_gpt_neox import LlavaGptNeoxForCausalLM
from llava.train.arguments_dataclass import ModelArguments, DataArguments, TrainingArguments
from llava.train.dataset import tokenizer_image_token


if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args.vision_tower = "openai/clip-vit-large-patch14-336"
    base_model = "llm-jp/llm-jp-1.3b-v1.0"

    # load model and tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model,
        model_max_length=1024,
        padding_side="right",
        use_fast=False,
    )
    if  model_args.base_model == "gpt2":
        model = LlavaGpt2ForCausalLM.from_pretrained(base_model)
    elif model_args.base_model == "gpt_neox":
        model = LlavaGptNeoxForCausalLM.from_pretrained(base_model)

    model.get_model().initialize_vision_modules(
        model_args=model_args,
    )

    model.load_state_dict(torch.load("./output_llava/checkpoints/finetuning-llava-v1.5-llm-jp-bf-llava-instruct-visual-genome/pytorch_model.pth"))
    model = model.to("cuda")
    model.to(torch.bfloat16)
    model.eval()

    conv_mode = "v1"
    conv = conv_templates[conv_mode].copy()

    # image pre-process
    image = Image.open("./imgs/sample1.jpg")
    image_tensor = model.get_model().vision_tower.image_processor(image, return_tensors='pt')['pixel_values'].half().cuda().to(torch.bfloat16)

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
    ).unsqueeze(0).cuda()
    input_ids = input_ids[:, :-1] # </sep>がinputの最後に入るので削除する
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=20.0)

    # predict
    with torch.inference_mode():
        output_ids = model.generate(
            inputs=input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.01,
            top_p=1.0,
            max_new_tokens=256,
            streamer=streamer,
            use_cache=True,
        )
        output_ids = [token_id for token_id in output_ids.tolist()[0] if token_id != IMAGE_TOKEN_INDEX]
        output = tokenizer.decode(output_ids)
        print(output)
        