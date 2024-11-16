"""
Utility functions for generic torch tensor operations.
"""

import torch


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
    ars2 = [torch.tensor(i, device=device).unsqueeze(0) for i in ars]
    ar = torch.cat(ars2).to(dtype=dtype)
    for i in ars2:
        del i
    return ar
