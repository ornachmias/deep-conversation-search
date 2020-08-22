from abc import ABC, abstractmethod


class BaseEncoder(ABC):
    def __init__(self, name):
        self.name = name
        
    @abstractmethod
    def encode(self, text):
        pass
