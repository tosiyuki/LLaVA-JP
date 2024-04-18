import gradio as gr
import torch
import transformers

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.model.llava_gpt2 import LlavaGpt2ForCausalLM
from llava.train.arguments_dataclass import ModelArguments, DataArguments, TrainingArguments
from llava.train.dataset import tokenizer_image_token


# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.bfloat16 if device=="cuda" else torch.float32
parser = transformers.HfArgumentParser(
    (ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()
#model_path = 'toshi456/llava-jp-1.3b-v1.0'
model_path = 'output_llava/checkpoints/finetune-llava-jp-1.3b-v1.1-siglip-so400m-patch14-384'

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


@torch.inference_mode()
def inference_fn(
    image, 
    prompt, 
    max_len, 
    temperature,
    repetition_penalty, 
    top_p, 
    ):
    # prepare inputs
    # image pre-process
    image_size = model.get_model().vision_tower.image_processor.size["height"]
    if model.get_model().vision_tower.scales is not None:
        image_size = model.get_model().vision_tower.image_processor.size["height"] * len(model.get_model().vision_tower.scales)

    if device == "cuda":
        image_tensor = model.get_model().vision_tower.image_processor(
            image, 
            return_tensors='pt', 
            size={"height": image_size, "width": image_size}
        )['pixel_values'].half().cuda().to(torch_dtype)
    else:
        image_tensor = model.get_model().vision_tower.image_processor(
            image, 
            return_tensors='pt', 
            size={"height": image_size, "width": image_size}
        )['pixel_values'].to(torch_dtype)

    # create prompt
    inp = DEFAULT_IMAGE_TOKEN + '\n' + prompt
    conv = conv_templates[conv_mode].copy()
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

    # generate
    output_ids = model.generate(
            inputs=input_ids,
            images=image_tensor,
            do_sample= temperature != 0.0,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_len,
            repetition_penalty=repetition_penalty,
            use_cache=True,
        )
    output_ids = [token_id for token_id in output_ids.tolist()[0] if token_id != IMAGE_TOKEN_INDEX]
    output = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    target = "システム: "
    idx = output.find(target)
    output = output[idx+len(target):]

    return output

with gr.Blocks() as demo:
    gr.Markdown(f"# LLaVA-JP Demo")

    with gr.Row():
        with gr.Column():
            # input_instruction = gr.TextArea(label="instruction", value=DEFAULT_INSTRUCTION)
            input_image = gr.Image(type="pil", label="image")
            prompt = gr.Textbox(label="prompt (optional)", value="")
            with gr.Accordion(label="Configs", open=False):
                max_len = gr.Slider(
                    minimum=10,
                    maximum=256,
                    value=50,
                    step=5,
                    interactive=True,
                    label="Max New Tokens",
                )
                
                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.1,
                    step=0.1,
                    interactive=True,
                    label="Temperature",
                )
            
                top_p = gr.Slider(
                    minimum=0.5,
                    maximum=1.0,
                    value=0.9,
                    step=0.1,
                    interactive=True,
                    label="Top p",
                )
        
                repetition_penalty = gr.Slider(
                    minimum=-1,
                    maximum=3,
                    value=1,
                    step=0.2,
                    interactive=True,
                    label="Repetition Penalty",
                )
            # button
            input_button = gr.Button(value="Submit")
        with gr.Column():
            output = gr.Textbox(label="Output")
    
    inputs = [input_image, prompt, max_len, temperature, repetition_penalty, top_p]
    input_button.click(inference_fn, inputs=inputs, outputs=[output])
    prompt.submit(inference_fn, inputs=inputs, outputs=[output])
    img2txt_examples = gr.Examples(examples=[[
        "./imgs/sample1.jpg",
        "猫の隣には何がありますか？",
        32,
        0.1,
        1.0,
        0.9,
    ],
    [
        "./imgs/sample2.jpg",
        "この画像の面白い点を教えてください？",
        256,
        0.1,
        1.0,
        0.9,
    ],
    ], inputs=inputs)
    
    
if __name__ == "__main__":
    demo.queue().launch()