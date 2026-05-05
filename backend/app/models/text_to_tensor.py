import numpy as np
import torch


def text_to_tensor(text: str, max_len: int = 500):
    """
    Convert text into a fixed numeric tensor.
    """

    encoded = [ord(c) for c in text[:max_len]]

    if len(encoded) < max_len:
        encoded += [0] * (max_len - len(encoded))

    array = np.array(encoded, dtype=np.float32)

    return torch.tensor(array).unsqueeze(0).unsqueeze(0)