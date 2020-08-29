import os
import pickle

import numpy as np
import matplotlib.pyplot as plt


class Visualization:
    def __init__(self, data_root):
        self.output_dir = os.path.join(data_root, 'visualization')
        os.makedirs(self.output_dir, exist_ok=True)

        self.results_dir = os.path.join(data_root, 'results')
        if not os.path.exists(self.results_dir):
            raise Exception('{} does not exists, please run experiments before visualization.'.format(self.results_dir))

        self.results = self.get_result_files()
        self.metric_visualize = 'f1'

    def visualize(self):
        encoders = ['bert', 'bow']
        distance_functions = ['cosine', 'l2']

        for encoder in encoders:
            for distance_function in distance_functions:
                plot_data = self.collect_results(encoder, distance_function)
                self.draw_plot(plot_data, encoder, distance_function)

    def collect_results(self, encoder, distance_function):
        datasets = ['reddit', 'cran']
        conversation_lengths = {
            's': 3,
            'm': 5,
            'l': 7,
            'xl': 10
        }

        window_sizes = [1, 2, 3, 4]
        graph_data = {}

        for dataset in datasets:
            graph_data[dataset] = {}
            graph_data[dataset]['conv_length'] = []

            filters = {'encoder': encoder, 'distance_func': distance_function, 'dataset': dataset}
            conv_length_grouping = self.group_results_by('conv_size', filters)
            for conv_length in conversation_lengths:
                f1_scores = []
                for r in conv_length_grouping[conv_length]:
                    r_loaded = self.load_result(r['path'])
                    f1_scores.append(np.mean([i[self.metric_visualize] for i in r_loaded[0]]))

                f1_mean = np.mean(f1_scores)
                conv_length_num = conversation_lengths[conv_length]
                graph_data[dataset]['conv_length'].append((conv_length_num, f1_mean))

            graph_data[dataset]['window_size'] = []
            window_size_grouping = self.group_results_by('window_size', filters)
            for window_size in window_sizes:
                f1_scores = []
                for r in window_size_grouping[str(window_size)]:
                    r_loaded = self.load_result(r['path'])
                    f1_scores.append(np.mean([i[self.metric_visualize] for i in r_loaded[0]]))

                f1_mean = np.mean(f1_scores)
                graph_data[dataset]['window_size'].append((window_size, f1_mean))

        return graph_data

    def draw_plot(self, plot_data, encoder, distance_function):
        fig = plt.figure(figsize=(8, 3))

        # One row, 2 columns
        ax1 = fig.add_subplot(121)

        ax1.plot(*zip(*plot_data['reddit']['conv_length']), marker='o', color='orange', label="Reddit")
        ax1.plot(*zip(*plot_data['cran']['conv_length']), marker='o', color='skyblue', label="CRAN")
        ax1.set_xlabel('Conversation Length')
        ax1.set_ylabel('F1 Score')
        ax1.legend()
        plt.draw()

        ax2 = fig.add_subplot(122)
        ax2.plot(*zip(*plot_data['reddit']['window_size']), marker='o', color='orange', label="Reddit")
        ax2.plot(*zip(*plot_data['cran']['window_size']), marker='o', color='skyblue', label="CRAN")
        ax2.set_xlabel('Window Size')
        ax2.set_ylabel('F1 Score')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, '{}_{}.png'.format(encoder, distance_function)))

    def get_result_files(self):
        results = []
        result_files = [f for f in
                        os.listdir(self.results_dir)
                        if os.path.isfile(os.path.join(self.results_dir, f))]

        for result_file in result_files:
            path = os.path.join(self.results_dir, result_file)
            name = result_file.replace('.pkl', '')
            params = name.split('_')
            curr_result = {
                'path': path,
                'dataset': params[0],
                'encoder': params[1],
                'conv_size': params[3],
                'window_size': params[4],
                'distance_func': params[5]
            }
            results.append(curr_result)

        return results

    def group_results_by(self, category, filters=None):
        if category not in ['dataset', 'encoder', 'conv_size', 'window_size', 'distance_func']:
            raise Exception('Invalid category name selected.')

        grouped_result = {}
        for r in self.results:
            is_filtered = False
            if filters is not None:
                for f in filters:
                    if r[f] == filters[f]:
                        is_filtered = True
                        break

            if not is_filtered:
                continue

            value = r[category]
            if value not in grouped_result:
                grouped_result[value] = []

            grouped_result[value].append(r)

        return grouped_result

    def load_result(self, path):
        return pickle.load(open(path, 'rb'))


visualization = Visualization('./data')
visualization.visualize()
