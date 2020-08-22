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

            evaluation = self.evaluate(conversation['correct_line_ids'], retrieved_msg_ids)
            result.append({
                'precision': evaluation.precision(),
                'recall': evaluation.recall(),
                'f1': evaluation.f1_score()
            })

        return result, index_time, np.mean(search_times)

    def search(self, conversation_id, query):
        start_time = time.time()
        result_filename = self.searcher.search(query, conversation_id)
        end_time = time.time() - start_time
        conv_id, msg_ids = Searcher.filename_to_ids(result_filename)
        return msg_ids, end_time

    def index(self):
        start_time = time.time()
        self.indexer.index()
        return time.time() - start_time

    def evaluate(self, relevant_msg_ids, retrieved_msg_ids):
        window_size = self.indexer.window_size
        evaluation = EvaluationMetrics(retrieved_msg_ids, relevant_msg_ids, window_size)
        return evaluation
