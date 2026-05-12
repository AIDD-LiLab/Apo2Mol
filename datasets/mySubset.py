from torch.utils.data import Dataset
from typing import (
    Sequence,
    TypeVar,
)

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')

class GetSubset(Dataset[T_co]):
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset, indices) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        data = self.dataset[self.indices[idx]]

        return data

    def __len__(self):
        return len(self.indices)
