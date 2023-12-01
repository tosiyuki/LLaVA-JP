import json
from pathlib import Path

import matplotlib.pyplot as plt

if __name__ == '__main__':
    train_data_paths = [
        './output_llava_llm_jp/finetuning-llava-v1.5-llm-jp-bf',
    ]

    train_losses = []
    for data_path in train_data_paths:
        with open(Path(data_path, 'trainer_state.json')) as f:
            log_data = f.read()
            log_data_json = json.loads(log_data)

        train_losses.append([history['loss'] for history in log_data_json['log_history'][:-1]])

    for i, train_loss in enumerate(train_losses):
        plt.plot(train_loss, label=str(i+1), linewidth=0.5)
    
    plt.title('Pretrain Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()