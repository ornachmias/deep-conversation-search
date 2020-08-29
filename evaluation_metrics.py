class EvaluationMetrics:
    def __init__(self, retrieved_msg_ids, relevant_msg_ids, top_msg_ids):
        self.retrieved_msg_ids = retrieved_msg_ids
        self.relevant_msg_ids = relevant_msg_ids
        self.top_msg_ids = top_msg_ids

    def precision(self):
        a = len(EvaluationMetrics.intersection(self.relevant_msg_ids, self.top_msg_ids))
        b = len(self.top_msg_ids)
        return a / b

    def recall(self):
        a = len(EvaluationMetrics.intersection(self.relevant_msg_ids, self.top_msg_ids))
        b = len(self.relevant_msg_ids)
        return a / b

    def f1_score(self):
        p = self.precision()
        r = self.recall()
        if p + r == 0:
            return 0
        return (2 * p * r) / (p + r)

    def precision_5(self):
        a = len(EvaluationMetrics.intersection(self.relevant_msg_ids, self.retrieved_msg_ids))
        b = len(self.retrieved_msg_ids)
        return a / b

    def recall_5(self):
        a = len(EvaluationMetrics.intersection(self.relevant_msg_ids, self.retrieved_msg_ids))
        b = len(self.relevant_msg_ids)
        return a / b

    def f1_score_5(self):
        p = self.precision_5()
        r = self.recall_5()
        if p + r == 0:
            return 0
        return (2 * p * r) / (p + r)

    @staticmethod
    def intersection(lst1, lst2):
        return list(set(lst1) & set(lst2))
