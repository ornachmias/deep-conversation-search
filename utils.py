from scipy.spatial import distance


class Utils:
    @staticmethod
    def distance(vec1, vec2):
        return distance.cosine(vec1, vec2)