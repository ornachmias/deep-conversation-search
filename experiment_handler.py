import numpy as np
import time

from datasets.base_dataset import BaseDataset
from evaluation_metrics import EvaluationMetrics
from indexer import Indexer
from searcher import Searcher
from tqdm import tqdm


class ExperimentHandler:
    def __init__(self, data_root, dataset: BaseDataset, indexer: Indexer, searcher: Searcher):
        self.data_root = data_root
        self.dataset = dataset
        self.indexer = indexer
        self.searcher = searcher

    def run(self):
        result = []
        index_time = self.index()
        search_times = []
        conversation_ids = self.dataset.get_conversation_ids()
        for conversation_id in tqdm(conversation_ids):
            conversation = self.dataset.get_conversation(conversation_id)
            query = conversation['query']
            retrieved_msg_ids, search_time = self.search(conversation_id, query)
            search_times.append(search_time)

            if not conversation['correct_line_ids']:
                continue

            evaluation = self.evaluate(conversation['correct_line_ids'], retrieved_msg_ids)
            result.append({
                'precision': evaluation.precision(),
                'recall': evaluation.recall(),
                'f1': evaluation.f1_score(),
                'precision@5': evaluation.precision_5(),
                'recall@5': evaluation.recall_5(),
                'f1@5': evaluation.f1_score_5()
            })

        return result, index_time, np.mean(search_times)

    def search(self, conversation_id, query):
        result = []
        start_time = time.time()
        top_results = self.searcher.search(query, conversation_id)
        end_time = time.time() - start_time
        for filename in top_results:
            conv_id, msg_ids = Searcher.filename_to_ids(filename)
            result.append([msg_ids])

        return result, end_time

    def index(self):
        start_time = time.time()
        self.indexer.index()
        return time.time() - start_time

    def evaluate(self, relevant_msg_ids, top_retrieved_msg_ids):
        # Flatten retrieved docs
        retrieved_msg_ids = set()
        for msg_ids in top_retrieved_msg_ids:
            retrieved_msg_ids.update(msg_ids[0])

        evaluation = EvaluationMetrics(list(retrieved_msg_ids), relevant_msg_ids, top_retrieved_msg_ids[0][0])
        return evaluation
