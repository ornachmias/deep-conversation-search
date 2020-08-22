import torch
from transformers import *

from encoders.base_encoder import BaseEncoder


class BertEncoder(BaseEncoder):
    def __init__(self):
        super(BertEncoder, self).__init__('bert_encoder')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        self.model.eval()

    def encode(self, text):
        input_ids = torch.LongTensor(self.tokenizer.encode(text)).to(self.device)
        input_ids = input_ids.unsqueeze(0)
        with torch.no_grad():
            out = self.model(input_ids=input_ids)
            hidden_states = out[2]
            # The BERT paper discusses how they reached the best
            # results by concatenating the output of the last four layers
            last_four_layers = [hidden_states[i] for i in (-1, -2, -3, -4)]
            cat_hidden_states = torch.cat(tuple(last_four_layers), dim=-1)
            cat_sentence_embedding = torch.mean(cat_hidden_states, dim=1).squeeze()
            return cat_sentence_embedding
