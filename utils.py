from scipy.spatial import distance
import numpy as np


class Utils:
    @staticmethod
    def cosine(vec1, vec2):
        return distance.cosine(vec1, vec2)

    @staticmethod
    def l2_norm(vec1, vec2):
        dist = np.linalg.norm(vec1 - vec2)