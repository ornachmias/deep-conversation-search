import numpy as np
from transformers import BertTokenizer

from encoders.base_encoder import BaseEncoder


class BowEncoder(BaseEncoder):
    def __init__(self):
        super(BowEncoder, self).__init__('bow_encoder')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self._vocab_size = 30522

    def encode(self, text):
        input_ids = self.tokenizer.encode(text)
        vec = np.zeros(self._vocab_size)
        for i in input_ids:
            vec[i] += 1

        return vec

