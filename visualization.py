import os
import pickle


class Visualization:
    def __init__(self, data_root):
        self.output_dir = os.path.join(data_root, 'visualization')
        os.makedirs(self.output_dir, exist_ok=True)

        self.results_dir = os.path.join(data_root, 'results')
        if not os.path.exists(self.results_dir):
            raise Exception('{} does not exists, please run experiments before visualization.'.format(self.results_dir))

        self.results = self.get_result_files()

    def visualize(self):
        results_by_dataset = self.group_results_by('dataset')

    def get_result_files(self):
        results = []
        result_files = [f for f in
                        os.listdir(self.results_dir)
                        if os.path.isfile(os.path.join(self.results_dir, f))]

        for result_file in result_files:
            path = os.path.join(self.results_dir, result_file)
            name = result_file.replace('.pkl', '')
            params = name.split('_')
            result = {
                'path': path,
                'dataset': params[0],
                'encoder': params[1],
                'conv_size': params[3],
                'window_size': params[4],
                'distance_func': params[5]
            }
            results.append(result)

        return results

    def group_results_by(self, category):
        if category not in ['dataset', 'encoder', 'conv_size', 'window_size', 'distance_func']:
            raise Exception('Invalid category name selected.')

        grouped_result = {}
        for r in self.results:
            value = r[category]
            if value not in grouped_result:
                grouped_result[value] = []

            grouped_result[value].append(r)

        return grouped_result

    def load_result(self, path):
        return pickle.load(open(path, 'rb'))



visualization = Visualization('./data')
result = visualization.load_result('./data/results/cran_bert_encoder_l_1_cosine.pkl')
print(result)