from abc import ABC, abstractmethod


class BaseDataset(ABC):
    def __init__(self, name, conversation_length):
        self.name = name
        self.conversation_length = BaseDataset._parse_conversation_length(conversation_length)

    @abstractmethod
    def init(self):
        pass

    @abstractmethod
    def get_conversation(self, conversation_id):
        pass

    @abstractmethod
    def get_conversation_ids(self):
        pass

    @staticmethod
    def _parse_conversation_length(length_name):
        lengths = {
            's': 3,
            'm': 5,
            'l': 7,
            'xl': 10
        }

        conversation_length = length_name.lower()
        if conversation_length not in lengths:
            raise Exception('Invalid conversation_length argument')

        return lengths[conversation_length]
