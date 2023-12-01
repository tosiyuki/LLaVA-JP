import os
import shutil
import json
from pathlib import Path

if __name__ == '__main__':
    dataset_dir = './'
    marge_path1 = Path(dataset_dir, "LLaVA-CC3M-Pretrain-595K", 'chat_ja_calm2.json')
    marge_path2 = Path(dataset_dir, "LLaVA-Stair-Caption", 'llava_stair_caption.json')

    with marge_path1.open('r', encoding='utf-8') as f:
        caption_data = f.read()
        caption_data_json = json.loads(caption_data)

    with marge_path2.open('r', encoding='utf-8') as f:
        caption_data = f.read()
        caption_data_json.extend(json.loads(caption_data))

    chat_ja_path = Path(dataset_dir, 'llava_pretrain_stair.json')
    with open(chat_ja_path, mode="w") as f:
        json.dump(caption_data_json, f, indent=2, ensure_ascii=False)
    