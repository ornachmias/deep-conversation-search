import os
import pickle
import re

from tqdm import tqdm

from encoders.base_encoder import BaseEncoder


class Searcher:
    def __init__(self, indexer_dir, encoder: BaseEncoder, top_results, comparison_func):
        self.indexer_dir = indexer_dir
        self.encoder = encoder
        self.top_results = top_results
        self.comparison_func = comparison_func

    def search(self, text, conversation_id):
        index_files = []
        for f in os.listdir(self.indexer_dir):
            if os.path.isfile(os.path.join(self.indexer_dir, f)) and Searcher.filename_to_ids(f)[0] == conversation_id:
                index_files.append(os.path.join(self.indexer_dir, f))

        result = {}
        text_encoded = self.encoder.encode(text)
        for index_file in tqdm(index_files):
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

    @staticmethod
    def filename_to_ids(filename):
        re_query = 'conv_(\\d\\d\\d\\d\\d)_msgs_(\\d\\d\\d\\d\\d)_(\\d\\d\\d\\d\\d)'
        m = re.search(re_query, filename)
        if m:
            return int(m.group(1)), list(range(int(m.group(2)), int(m.group(3))))

        return None
