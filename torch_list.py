"""List serialized into a pytorch tensor.

Pytorch uses python multiprocessing when using multiple workers for the dataloader. Using a python
list to store a bunch of image paths will lead to OOMs because every worker will slowly duplicate
the lists in the dataloader. The memory efficiency can be improved with numpy, but most of the
memory will still be duplicated between threads. With a torch tensor this can be avoided.
"""
from typing import List
import pickle
import torch
import numpy as np

class TorchList:
    """Python list stored as serialized objects in a torch tensor.

    Using this in data loaders will avoid memory copies between the dataloader workers.
    """
    def __init__(self, python_list: List):
        serialized_list = [pickle.dumps(d, protocol=-1) for d in python_list]
        addresses = [len(s) for s in serialized_list]

        # Use np.frombuffer instead of torch.frombuffer directly to avoid warnings about nonwritable
        # buffers or arrays.
        serialized_list = [np.frombuffer(s, dtype=np.uint8) for s in serialized_list]
        serialized_list = np.concatenate(serialized_list)
        self._serialized_list = torch.from_numpy(serialized_list)

        self._addresses = torch.tensor(addresses, dtype=torch.int64)
        self._addresses = torch.cumsum(self._addresses, dim=-1)

    def __len__(self):
        return len(self._addresses)

    def __getitem__(self, index):
        if index == 0:
            start_address = 0
        else:
            start_address = int(self._addresses[index - 1])
        end_address = int(self._addresses[index])
        list_entry_bytes = memoryview(self._serialized_list[start_address:end_address].numpy())
        return pickle.loads(list_entry_bytes)


# Test the pytorch list
if __name__ == "__main__":
    import os
    some_paths = list(os.listdir("/home/willem/"))
    some_torch_paths = TorchList(some_paths)

    for i, pp in enumerate(some_paths):
        tp = some_torch_paths[i]
        assert(tp == pp)

    print("Success")
