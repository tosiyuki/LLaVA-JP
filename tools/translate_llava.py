"""LLaVA-CC3M-Pretrain-595Kの日本語用データを作成する."""
import json
import gc
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer


def build_calm_prompt(input: str) -> str:
    input = input.replace(" .", ".")
    input = input.replace(" !", "!")
    input = input.replace(" ?", "?")
    input = input.replace(" ,", ",")
    prompt = f"""USER: 下記の英語を日本語に翻訳してください。
{input}
ASSISTANT: """
    
    return prompt


if __name__ == '__main__':
    instruction_dic = {
        "Create a compact narrative representing the image presented.\n<image>": "与えられた画像を表す簡潔な文を作成してください。\n<image>",
        "Describe the image concisely.\n<image>": "画像について簡単に説明してください。\n<image>",
        "Provide a brief description of the given image.\n<image>": "与えられた画像について簡単に説明してください。\n<image>",
        "Offer a succinct explanation of the picture presented.\n<image>": "入力された写真について簡単に説明してください。\n<image>",
        "Summarize the visual content of the image.\n<image>": "画像の内容を教えてください。\n<image>",
        "Give a short and clear explanation of the subsequent image.\n<image>": "次の画像を短く分かりやすく説明してください。\n<image>",
        "Share a concise interpretation of the image provided.\n<image>": "与えられた画像について教えてください。\n<image>",
        "Present a compact description of the photo\'s key features.\n<image>": "写真の特徴を手短に教えてください。\n<image>",
        "Relay a brief, clear account of the picture shown.\n<image>": "この画像について短い言葉で説明してください。\n<image>",
        "Render a clear and concise summary of the photo.\n<image>": "写真の概要を簡潔かつ分かりやすく伝えてください。\n<image>",
        "Write a terse but informative summary of the picture.\n<image>": "写真について簡単に説明してください。\n<image>",

        "<image>\nCreate a compact narrative representing the image presented.": "<image>\n与えられた画像を表す簡潔な文を作成してください。",
        "<image>\nDescribe the image concisely.": "<image>\n画像について簡単に説明してください。",
        "<image>\nProvide a brief description of the given image.": "<image>\n与えられた画像について簡単に説明してください。",
        "<image>\nOffer a succinct explanation of the picture presented.": "<image>\n入力された写真について簡単に説明してください。",
        "<image>\nSummarize the visual content of the image.": "<image>\n画像の内容を教えてください。",
        "<image>\nGive a short and clear explanation of the subsequent image.": "<image>\n次の画像を短く分かりやすく説明してください。",
        "<image>\nShare a concise interpretation of the image provided.": "<image>\n与えられた画像について教えてください。",
        "<image>\nPresent a compact description of the photo\'s key features.": "<image>\n写真の特徴を手短に教えてください。",
        "<image>\nRelay a brief, clear account of the picture shown.": "<image>\nこの画像について短い言葉で説明してください。",
        "<image>\nRender a clear and concise summary of the photo.": "<image>\n写真の概要を簡潔かつ分かりやすく伝えてください。",
        "<image>\nWrite a terse but informative summary of the picture.": "<image>\n写真について簡単に説明してください。",
    }

    INPUT_DIR = 'input/LLaVA-CC3M-Pretrain-595K'
    chat_path = Path(INPUT_DIR, 'chat.json')
    
    with chat_path.open('r', encoding='utf-8') as f:
        chat_data = f.read()
        chat_data_json = json.loads(chat_data)

    # チェックポイントの読み込み
    model = AutoModelForCausalLM.from_pretrained("cyberagent/calm2-7b-chat", device_map="auto", torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained("cyberagent/calm2-7b-chat")

    instruction = "次の英語を日本語に翻訳してください。"

    try:
        for chat in tqdm(chat_data_json, total=len(chat_data_json)):
            prompt = build_calm_prompt(chat['conversations'][1]['value'])
            token_ids = tokenizer.encode(prompt, return_tensors="pt")
            output_ids = model.generate(
                input_ids=token_ids.to(model.device),
                max_new_tokens=512,
                do_sample=True,
                temperature=0.01,
            )
            output_ids = output_ids.to("cpu")
            translation_text = tokenizer.decode(output_ids.tolist()[0])
            match = re.search(r'ASSISTANT: (.*)', translation_text)
            if match:
                translation_text = match.group(1).strip()
            else:
                translation_text = translation_text.replace(prompt, "")
                translation_text = translation_text.replace("</s>", "")
            translation_text = translation_text.replace("<|endoftext|>", "")
            translation_text = translation_text.strip()
            
            # インストラクションデータの置き換え
            chat['conversations'][0]['from'] = 'ユーザー'
            chat['conversations'][0]['value'] = instruction_dic[chat['conversations'][0]['value']]
            chat['conversations'][1]['from'] = 'システム'
            chat['conversations'][1]['value'] = translation_text

            del prompt, token_ids, output_ids, translation_text
            gc.collect()
            torch.cuda.empty_cache()

        chat_ja_path = Path(INPUT_DIR, 'chat_ja_calm2.json')
        with open(chat_ja_path, mode="w") as f:
            json.dump(chat_data_json, f, indent=2, ensure_ascii=False)
    except:
        print('例外発生')
        chat_ja_path = Path(INPUT_DIR, 'chat_ja_calm2.json')
        with open(chat_ja_path, mode="w") as f:
            json.dump(chat_data_json, f, indent=2, ensure_ascii=False)
