"""
Utility functions for generic torch tensor operations.
"""

import numpy as np
import torch


def round_mantissa(arr: np.ndarray, keep_bits: int) -> np.ndarray:
    """
    Zero the low mantissa bits of a float16 array (round-to-nearest), keeping ``keep_bits`` of the 10 mantissa
    bits. Lossy, but the model consumes activations under bf16/f16 autocast and averages over runs, so the low
    bits are noise to it: on real data, keeping 4 bits leaves the predicted Hi-C map correlated 0.999998 with
    the full-precision map while making the (already compressed) cache ~1.7x smaller. ``keep_bits >= 10``
    returns an unchanged copy.

    The trick is a single integer op on the float16 bit pattern (sign|exp|mantissa): add half of the lowest
    kept bit, then clear the dropped low bits -- rounding carries into the exponent correctly because the
    mantissa is adjacent to it.
    """
    arr = np.asarray(arr)
    if arr.dtype != np.float16:
        raise ValueError(f"round_mantissa expects a float16 array, got {arr.dtype}")
    drop = 10 - int(keep_bits)
    if drop <= 0:
        return arr.copy()
    u = arr.view(np.uint16).astype(np.uint32)
    u = (u + (1 << (drop - 1))) & np.uint32(0xFFFF - ((1 << drop) - 1))  # round half up, clear low `drop` bits
    return (u & 0xFFFF).astype(np.uint16).view(np.float16)


def list_to_tensor_batch(ars, device, dtype=torch.float32):
    """
    Converts a list of arrays into a batch tensor, performing concatenation on the device for speed.

    This function is useful when you have a list of very large numpy arrays that you want to convert to a tensor.
    Instead of concatenating and converting dtypes in numpy (slow), this function converts each array to a tensor "as
    is" and concatenates them on the specified device. A fairly common usecase would include converting and
    transferring a list of float16 arrays to a GPU and then converting them to float32, thus avoiding the overhead of
    transferring the 2x larger arrays to the GPU, and slow concatenation on the CPU.


    Parameters
    ----------
    ars : list
        List of arrays or data that can be converted to tensors.
    device : torch.device or str
        The device to which the tensors will be moved (e.g., 'cpu' or 'cuda').
    dtype : torch.dtype, optional
        Desired data type of the resulting tensor. Default is torch.float32.

    Returns
    -------
    torch.Tensor
        Concatenated tensor containing all input arrays, with an added dimension,
        moved to the specified device and converted to the specified dtype.
    """
    if isinstance(ars[0], torch.Tensor):
        return torch.stack(ars).to(device=device, dtype=dtype)
    ars2 = [torch.tensor(i, device=device).unsqueeze(0) for i in ars]
    ar = torch.cat(ars2).to(dtype=dtype)
    for i in ars2:
        del i
    return ar
