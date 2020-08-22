class EvaluationMetrics:
    def __init__(self, retrieved_msg_ids, relevant_msg_ids, window_size):
        self.retrieved_msg_ids = retrieved_msg_ids
        self.relevant_msg_ids = relevant_msg_ids
        self.window_size = window_size

    def precision(self):
        a = len(EvaluationMetrics.intersection(self.relevant_msg_ids, self.retrieved_msg_ids))
        b = len(self.retrieved_msg_ids)
        return a / b

    def recall(self):
        return len(EvaluationMetrics.intersection(self.relevant_msg_ids, self.retrieved_msg_ids)) / self.window_size

    def f1_score(self):
        p = self.precision()
        r = self.recall()
        return (2 * p * r) / (p + r)

    @staticmethod
    def intersection(lst1, lst2):
        return list(set(lst1) & set(lst2))
