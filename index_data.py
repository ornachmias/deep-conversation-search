import argparse

from datasets.cran_dataset import CranDataset
from datasets.reddit_dataset import RedditDataset
from encoders.bert_encoder import BertEncoder
from encoders.bow_encoder import BowEncoder
from indexer import Indexer

parser = argparse.ArgumentParser(description='Deep Conversation Search Data Indexer')
parser.add_argument('-d', '--data_dir', default='./data')
args = parser.parse_args()

data_root = args.data_dir
conversation_lengths = ['s', 'm', 'l', 'xl']
window_sizes = [1, 2, 3, 4]
encoders = [BertEncoder(), BowEncoder()]

for encoder in encoders:
    for window_size in window_sizes:
        for conversation_length in conversation_lengths:
            datasets = [RedditDataset(data_root, conversation_length), CranDataset(data_root, conversation_length)]
            for dataset in datasets:
                print('Current indexer parameters: encoder={}, window_size={}, conversation_length={}, dataset={}'
                      .format(encoder.name, window_size, conversation_length, dataset.name))
                dataset.init()
                indexer = Indexer(data_root, encoder, dataset, window_size)
                indexer.index()
