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


parser = argparse.ArgumentParser(description='Deep Conversation Search Analyze')
parser.add_argument('-d', '--data_dir', default='./data')
parser.add_argument('-s', '--dataset', type=str.lower, choices=['reddit', 'cran'], default='cran')
parser.add_argument('-e', '--encoder', type=str.lower, choices=['bert', 'bow'], default='bow')
parser.add_argument('-l', '--conversation_length', choices=['s', 'm', 'l', 'xl'], default='s')
parser.add_argument('-w', '--window_size', type=int, default=3)
parser.add_argument('-c', '--compare_func', type=str.lower, choices=['cosine', 'l2'], default='cosine')
args = parser.parse_args()

data_root = args.data_dir
if args.encoder == 'bert':
    encoder = BertEncoder()
elif args.encoder == 'bow':
    encoder = BowEncoder()
else:
    raise Exception('Invalid encoder selected.')

if args.compare_func == 'cosine':
    compare_func = Utils.cosine
elif args.compare_func == 'l2':
    compare_func = Utils.l2_norm
else:
    raise Exception('Invalid compare_func selected.')

if args.dataset == 'reddit':
    dataset = RedditDataset(data_root=data_root, conversation_length=args.conversation_length)
elif args.dataset == 'cran':
    dataset = CranDataset(data_root=data_root, conversation_length=args.conversation_length)
else:
    raise Exception('Invalid dataset selected.')

dataset.init()
indexer = Indexer(data_root=data_root, encoder=encoder, dataset=dataset, window_size=args.window_size)
searcher = Searcher(indexer.indexer_dir, encoder=encoder, top_results=5, comparison_func=compare_func)
experiment = ExperimentHandler(data_root=data_root, dataset=dataset, indexer=indexer, searcher=searcher)

run_result = experiment.run()
results_dir = os.path.join(data_root, 'results')
os.makedirs(results_dir, exist_ok=True)

filename = '{}_{}_{}_{}_{}.pkl'.format(dataset.name, encoder.name, args.conversation_length.lower(),
                                       args.window_size, args.compare_func)
pickle.dump(run_result, open(os.path.join(results_dir, filename), 'wb'))
