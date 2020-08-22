import os
import pickle
from tqdm import tqdm

from encoders.base_encoder import BaseEncoder


class Searcher:
    def __init__(self, indexer_dir, encoder: BaseEncoder, top_results, comparison_func):
        self._index_files = [os.path.join(indexer_dir, f)
                             for f in
                             os.listdir(indexer_dir) if os.path.isfile(os.path.join(indexer_dir, f))]
        self.encoder = encoder
        self.top_results = top_results
        self.comparison_func = comparison_func

    def search(self, text):
        result = {}
        text_encoded = self.encoder.encode(text)
        for index_file in tqdm(self._index_files):
            file_name = os.path.basename(index_file).replace('.pkl', '')
            index = pickle.load(open(index_file, 'rb'))
            distance = self.comparison_func(text_encoded, index)

            if len(result) < self.top_results:
                result[file_name] = distance
            else:
                max_value = max(result, key=result.get)
                if result[max_value] > distance:
                    result.pop(max_value, None)
                    result[file_name] = distance

        return result
