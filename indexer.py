import os
import pickle
from tqdm import tqdm

from datasets.base_dataset import BaseDataset
from encoders.base_encoder import BaseEncoder


class Indexer:
    def __init__(self, data_root, encoder: BaseEncoder, dataset: BaseDataset, window_size):
        self.window_size = window_size
        self.encoder = encoder
        self.dataset = dataset

        self.indexer_dir = os.path.join(data_root, 'index',
                                        self.dataset.name, self.encoder.name,
                                        'window_{0:02d}'.format(self.window_size))
        os.makedirs(self.indexer_dir, exist_ok=True)

    def index(self):
        conversation_ids = self.dataset.get_conversation_ids()
        for conv_id in tqdm(conversation_ids):
            conversation = self.dataset.get_conversation(conv_id)['conversation']
            msgs_ids = list(conversation.keys())
            conv_size = len(msgs_ids)
            if conv_size <= self.window_size:
                continue

            for i in range(conv_size - self.window_size + 1):
                current_msg_ids = msgs_ids[i:i+self.window_size]
                content = Indexer._get_msgs_content(conversation, current_msg_ids)
                filename = 'conv_{0:05d}_msgs_{0:05d}_{0:05d}.pkl'.format(conv_id, i, i + self.window_size)
                file_path = os.path.join(self.indexer_dir, filename)
                if os.path.exists(file_path):
                    continue

                vec = self.encoder.encode(content)
                pickle.dump(vec, open(file_path, 'wb'))

    @staticmethod
    def _get_msgs_content(conversation, msg_ids):
        content = ''
        for msg_key in msg_ids:
            content += ' ' + conversation[msg_key]
        content = content.strip()
        return content
