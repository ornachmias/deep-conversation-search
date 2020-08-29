import argparse
import os
import pickle

from datasets.cran_dataset import CranDataset
from datasets.reddit_dataset import RedditDataset
from encoders.bert_encoder import BertEncoder
from encoders.bow_encoder import BowEncoder
from experiment_handler import ExperimentHandler
from indexer import Indexer
from searcher import Searcher
from utils import Utils

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
                print('Current experiment parameters: encoder={}, window_size={}, conversation_length={}, dataset={}'
                      .format(encoder.name, window_size, conversation_length, dataset.name))
                dataset.init()
                indexer = Indexer(data_root=data_root, encoder=encoder, dataset=dataset, window_size=window_size)
                searcher = Searcher(indexer.indexer_dir, encoder=encoder, top_results=5, comparison_func=Utils.l2_norm)
                experiment = ExperimentHandler(data_root=data_root, dataset=dataset, indexer=indexer, searcher=searcher)

                run_result = experiment.run()
                results_dir = os.path.join(data_root, 'results')
                os.makedirs(results_dir, exist_ok=True)

                filename = '{}_{}_{}_{}_{}.pkl'.format(dataset.name, encoder.name, conversation_length,
                                                       window_size, 'l2')
                pickle.dump(run_result, open(os.path.join(results_dir, filename), 'wb'))
