import re
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Set


class StrEncoder(metaclass=ABCMeta):
    _vocabulary: Dict[str, int] = {}

    def _get_or_create(self, key: str) -> int:
        result: Optional[int] = self._vocabulary.get(key)

        if result is not None:
            return result

        result = len(self._vocabulary)
        self._vocabulary[key] = result
        return result

    def vocabulary_size(self):
        return len(self._vocabulary)

    @abstractmethod
    def encode(self, value: str) -> Set[int]:
        pass


class ShingleEncoder(StrEncoder):
    _vocabulary: Dict[str, int] = {}
    _k: int

    def __init__(self, k: int):
        super().__init__()
        self._k = k

    def encode(self, value: str) -> Set[int]:
        k = self._k

        if len(value) < k:
            return set()
        elif len(value) == k:
            return {self._get_or_create(value)}

        return {
            self._get_or_create(value[i : (i + k)]) for i in range(len(value) - k + 1)
        }

    @staticmethod
    def shingle(value: str, k: int) -> Set[str]:
        if len(value) < k:
            return set()
        elif len(value) == k:
            return {value}

        return {value[i : (i + k)] for i in range(len(value) - k + 1)}


class ModelWordsEncoder(StrEncoder):
    def __init__(self):
        super().__init__()

    def encode(self, value: str) -> Set[int]:
        matches = ModelWordsEncoder.get_model_words(value)

        return {self._get_or_create(v) for v in matches}

    @staticmethod
    def get_model_words(value: str) -> List[str]:
        return re.findall(
            "[a-zA-Z0-9]*(([0-9]+[ˆ0-9, ]+)|([ˆ0-9, ]+[0-9]+))[a-zA-Z0-9]*", value
        )
