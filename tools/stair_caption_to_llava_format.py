import json
from pathlib import Path
from tqdm import tqdm


if __name__ == '__main__':
    #dataset_dir = './'
    dataset_dir = '/home/toshi/work/git/ImageCaptionZoo/dataset'
    stair_captions_dir = 'stair_captions_v1.2'
    caption_path = Path(dataset_dir, stair_captions_dir, 'stair_captions_v1.2_train.json')
    
    stair_llava_formats = []

    with caption_path.open('r', encoding='utf-8') as f:
        caption_data = f.read()
        caption_data_json = json.loads(caption_data)

    for i, image in tqdm(enumerate(caption_data_json['images']), total=len(caption_data_json['images'])):
        captions = [(annotation['id'], annotation['caption']) for annotation in caption_data_json['annotations'] if annotation['image_id'] == image['id']]

        if len(captions) == 0:
            print("not found caption")
            continue

        for id, caption in captions:
            llava_format = {}
            conversations = []

            conversation_user = {
                'from': 'ユーザー',
                'value': '画像について説明してください。\n<image>'
            }
            conversation_system = {
                'from': 'システム',
                'value': caption
            }
            conversations.append(conversation_user)
            conversations.append(conversation_system)

            llava_format['id'] = id
            llava_format['image'] = image['file_name']
            llava_format['conversations'] = conversations

            stair_llava_formats.append(llava_format)

    chat_ja_path = Path('dataset', 'llava_stair_caption.json')
    with open(chat_ja_path, mode="w") as f:
        json.dump(stair_llava_formats, f, indent=2, ensure_ascii=False)
    