import json
import os
import pickle
import random

from tqdm import tqdm

from datasets.base_dataset import BaseDataset


class RedditDataset(BaseDataset):
    def __init__(self, data_root, conversation_length='s'):
        super(RedditDataset, self).__init__('reddit', conversation_length)

        self._data_file = os.path.join(data_root, 'reddit', 'coarse_discourse_dump_reddit.json')
        cache_dir = os.path.join(data_root, 'cache')
        os.makedirs(cache_dir, exist_ok=True)

        self._cache_file = os.path.join(cache_dir, 'reddit_{}.pkl'.format(conversation_length))
        self.conversations = None

    def init(self):
        if os.path.exists(self._cache_file):
            self.conversations = pickle.load(open(self._cache_file, 'rb'))
            return

        raw_convs = []
        with open(self._data_file) as json_file:
            lines = json_file.readlines()
            for line in tqdm(lines):
                raw_convs.append(RedditDataset._parse_line(line))

        self.conversations = self._generate_conversations(raw_convs)
        pickle.dump(self.conversations, open(self._cache_file, 'wb'))

    def get_conversation_ids(self):
        return list(self.conversations.keys())

    def get_conversation(self, conversation_id):
        return self.conversations[conversation_id]

    def _generate_conversations(self, raw_conversations):
        conversations = {}
        available_indices = set(list(range(len(raw_conversations))))
        conv_id = 0
        while len(available_indices) > self.conversation_length:
            correct_line_ids = []
            generated_conv = {}
            query = None

            selected_indices = random.sample(list(available_indices), self.conversation_length)
            correct_index = random.choice(selected_indices)
            msg_id = 0

            for i in selected_indices:
                conv = raw_conversations[i]

                if i == correct_index:
                    query = conv['query']

                for msg in conv['conversation']:
                    generated_conv[msg_id] = msg
                    if i == correct_index:
                        correct_line_ids.append(msg_id)

                    msg_id += 1

                available_indices.remove(i)

            conversations[conv_id] = {
                'conversation': generated_conv,
                'correct_line_ids': correct_line_ids,
                'query': query
            }

            conv_id += 1

        return conversations

    @staticmethod
    def _parse_line(line):
        reader = json.loads(line)
        result = {
            'query': reader['title'],
            'conversation': []
        }

        for p in reader['posts']:
            if 'body' in p:
                result['conversation'].append(p['body'])

        return result
