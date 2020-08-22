import os
import pickle
import random


from datasets.base_dataset import BaseDataset


class CranDataset(BaseDataset):
    def __init__(self, data_root, conversation_length='s'):
        super(CranDataset, self).__init__('cran', conversation_length)

        self._queries_file = os.path.join(data_root, 'cran', 'cran.qry')
        self._documents_file = os.path.join(data_root, 'cran', 'cran.all.1400')
        self._ranking_file = os.path.join(data_root, 'cran', 'cranqrel')

        cache_dir = os.path.join(data_root, 'cache')
        os.makedirs(cache_dir, exist_ok=True)

        self._cache_file = os.path.join(cache_dir, 'cran_{}.pkl'.format(conversation_length))
        self.conversations = None

    def init(self):
        if os.path.exists(self._cache_file):
            self.conversations = pickle.load(open(self._cache_file, 'rb'))
            return

        rankings, docs, queries = self._parse_files()
        self.conversations = self._generate_conversations(rankings, docs, queries)
        pickle.dump(self.conversations, open(self._cache_file, 'wb'))

    def get_conversation_ids(self):
        return list(self.conversations.keys())

    def get_conversation(self, conversation_id):
        return self.conversations[conversation_id]

    def _parse_files(self):
        rankings = self._parse_ranking_file()
        docs = CranDataset._parse_general_file(self._documents_file)
        queries = CranDataset._parse_general_file(self._queries_file)
        return rankings, docs, queries

    def _generate_conversations(self, ranking, docs, queries):
        conversations = {}
        available_indices = set([r for r in ranking if r in queries])
        conv_id = 0

        while len(available_indices) > self.conversation_length and \
                len(set([ranking.get(key) for key in available_indices])) > self.conversation_length:
            correct_line_ids = []
            generated_conv = {}
            query = None

            selected_query_ids = random.sample(list(available_indices), self.conversation_length)
            correct_query_id = random.choice(selected_query_ids)
            msg_id = 0

            for query_id in selected_query_ids:
                conv = docs[ranking[query_id]].split('.')

                if query_id == correct_query_id:
                    query = queries[query_id]

                for msg in conv:
                    if msg is None or msg == '':
                        continue

                    generated_conv[msg_id] = msg
                    if query_id == correct_query_id:
                        correct_line_ids.append(msg_id)

                    msg_id += 1

                available_indices.remove(query_id)

            conversations[conv_id] = {
                'conversation': generated_conv,
                'correct_line_ids': correct_line_ids,
                'query': query
            }

            conv_id += 1

        return conversations

    @staticmethod
    def _parse_general_file(file_path):
        docs_dict = {}
        with open(file_path, 'r') as docs_file:
            start_append = False
            last_doc_id = None
            text = ''
            for doc_line in docs_file:
                if doc_line.startswith('.I '):
                    if last_doc_id is not None:
                        docs_dict[last_doc_id] = text
                        text = ''

                    doc_id = int(doc_line.replace('.I ', '').strip())
                    last_doc_id = doc_id
                    docs_dict[doc_id] = []
                    start_append = False

                if start_append:
                    text += ' ' + doc_line.strip().replace('\n', '')

                if doc_line.strip().startswith('.W'):
                    start_append = True

        return docs_dict

    def _parse_ranking_file(self):
        ranking_dict = {}
        with open(self._ranking_file, 'r') as ranking_file:
            for rank_line in ranking_file:
                query_id = int(rank_line.split(' ')[0])
                doc_id = rank_line.split(' ')[1]
                rank = rank_line.split(' ')[2]
                if rank is None or rank == '-1' or rank == '':
                    continue

                if query_id not in ranking_dict:
                    ranking_dict[query_id] = []

                ranking_dict[query_id].append((int(doc_id), int(rank)))

        for query_id in ranking_dict:
            selected_rank = None
            selected_doc_id = None
            for doc_id, rank in ranking_dict[query_id]:
                if selected_rank is None or selected_rank > rank:
                    selected_doc_id = doc_id

            ranking_dict[query_id] = selected_doc_id

        return ranking_dict
