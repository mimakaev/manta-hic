"""
Helper functions for Hi-C operations in PyTorch, not directly involved in the model.

TODO: Add tests for these functions.
"""

import math

import torch
import torch.nn.functional as F


# Helper functions
def _coarsen(ar: torch.Tensor, operation: str = "sum") -> torch.Tensor:
    """
    Coarsen an array by a factor of 2 along the last two dimensions
    using the specified operation: 'sum' or 'nanmin'
    """
    W, H = ar.shape[-1], ar.shape[-2]
    ar = ar.reshape(*ar.shape[:-2], H // 2, 2, W // 2, 2)  # Shape: [..., H/2, 2, W/2, 2]
    if operation == "sum":
        ar = ar.sum(dim=-3)
        ar = ar.sum(dim=-1)
    elif operation == "nanmin":
        large_num = 1e6  # Replace NaN values with a large number
        ar = ar.clone()
        ar[ar.isnan()] = large_num

        ar = ar.min(dim=-3)[0]
        ar = ar.min(dim=-1)[0]
        ar[ar == large_num] = float("nan")  # Restore NaN values
    else:
        raise ValueError("Unsupported operation")
    return ar  # Shape: [..., H/2, W/2]


def _expand(ar: torch.Tensor) -> torch.Tensor:
    """
    Expands an array by a factor of 2 along the last two dimensions
    """
    H, W = ar.shape[-2], ar.shape[-1]
    newar = ar.new_zeros(*ar.shape[:-2], H * 2, W * 2)
    newar[..., ::2, ::2] = ar
    newar[..., 1::2, ::2] = ar
    newar[..., ::2, 1::2] = ar
    newar[..., 1::2, 1::2] = ar
    return newar


# Main functions below


# Borrowed from cooltools.lib.numutils but rewrote in torch
@torch.no_grad()
def adaptive_coarsegrain_torch(
    ar: torch.Tensor,
    countar: torch.Tensor,
    cutoff: int = 5,
    max_levels: int = 8,
    min_shape: int = 8,
) -> torch.Tensor:
    """
    Adaptively coarsegrain a Hi-C matrix based on local neighborhood pooling of counts.

    This function performs adaptive coarsegraining of Hi-C matrices using PyTorch,
    based on the `adaptive_coarsegrain` method from the Open2C cooltools package.

    Parameters
    ----------
    ar : torch.Tensor
        A batch of square Hi-C matrices to coarsegrain. Shape [B, C, N, N].
    countar : torch.Tensor
        The raw count matrices for the same area. Must have the same shape as `ar`.
        Shape [B, C, N, N].
    cutoff : int, optional
        Minimum number of raw counts per pixel required to stop 2x2 pooling. Default is 5.
    max_levels : int, optional
        Maximum number of levels of coarsening to perform. Default is 8.
    min_shape : int, optional
        Stops coarsegraining when the coarsegrained array shape is less than this value. Default is 8.

    Returns
    -------
    torch.Tensor
        Coarsegrained Hi-C matrix. Shape [B, C, N, N].
    """

    # Ensure ar and countar are float tensors
    ar = ar.float()
    countar = countar.float()

    B, C, N, _ = ar.shape
    Norig = N

    Nlog = math.log2(Norig)
    if not math.isclose(Nlog, round(Nlog)):
        newN = int(2 ** math.ceil(Nlog))  # next power-of-two sized matrix
        # Now, we need to pad ar and countar to shape [B, C, newN, newN]
        newar = torch.full((B, C, newN, newN), float("nan"), dtype=ar.dtype, device=ar.device)
        newar[..., :Norig, :Norig] = ar
        ar = newar

        newcountar = torch.zeros((B, C, newN, newN), dtype=countar.dtype, device=countar.device)
        newcountar[..., :Norig, :Norig] = countar
        countar = newcountar

    # Mask of valid elements
    armask = torch.isfinite(ar)
    ar = ar.clone()
    ar[~armask] = 0
    countar = countar.clone()
    countar[~armask] = 0

    # Prepare lists for coarsened arrays
    ar_cg = [ar]
    countar_cg = [countar]
    armask_cg = [armask]

    # 1. Forward pass: coarsegrain all 3 arrays
    for _ in range(max_levels):
        H = countar_cg[-1].shape[-2]
        W = countar_cg[-1].shape[-1]
        if H > min_shape and W > min_shape:
            countar_cg.append(_coarsen(countar_cg[-1], operation="sum"))
            armask_cg.append(_coarsen(armask_cg[-1], operation="sum"))
            ar_cg.append(_coarsen(ar_cg[-1], operation="sum"))
        else:
            break

    # Get the most coarsegrained array
    ar_cur = ar_cg.pop()
    _ = countar_cg.pop()  # we only use the second coarsegrained array
    armask_cur = armask_cg.pop()

    # 2. Reverse pass: replace values starting with most coarsegrained array
    for _ in range(len(countar_cg)):
        ar_next = ar_cg.pop()
        countar_next = countar_cg.pop()
        armask_next = armask_cg.pop()

        # obtain current "average" value by dividing sum by the # of valid pixels
        val_cur = ar_cur / armask_cur
        # expand it so that it is the same shape as the previous level
        val_exp = _expand(val_cur)
        # create array of substitutions: multiply average value by counts
        addar_exp = val_exp * armask_next

        # make a copy of the raw Hi-C array at current level
        countar_next_mask = countar_next.clone()
        countar_next_mask[~armask_next.bool()] = float("nan")  # fill nans
        countar_coarsened = _coarsen(countar_next_mask, operation="nanmin")
        countar_exp = _expand(countar_coarsened)

        # replacement mask
        curmask = countar_exp < cutoff
        # procedure of replacement
        ar_next[curmask] = addar_exp[curmask]
        # setting zeros at invalid pixels
        ar_next[~armask_next.bool()] = 0

        # prepare for the next level
        ar_cur = ar_next
        armask_cur = armask_next

    # Final adjustments
    ar_next[~armask_next] = float("nan")
    ar_next = ar_next[..., :Norig, :Norig]

    return ar_next  # [B, C, N, N]


@torch.no_grad()
def create_expected_matrix(
    snippet: torch.Tensor,
    weight: torch.Tensor,
    exp: torch.Tensor,
) -> torch.Tensor:
    """
    Creates an expected matrix from weights and expected values, and adjusts the Hi-C snippet to match missing bins.

    This function modifies the given Hi-C snippet based on provided weights and expected values,
    while also generating a matching expected matrix.

    Parameters
    ----------
    snippet : torch.Tensor
        The input Hi-C maps. Shape [B, C, N, N].
    weight : torch.Tensor
        The weight tensor for each channel. Shape [B, C, N].
    exp : torch.Tensor
        The expected value tensor. Shape [B, C, N * 5 / 4].

    Returns
    -------
    tuple
        A tuple containing:
        - torch.Tensor: The modified Hi-C snippet tensor with zeros matching expected values. Shape [B, C, N, N].
        - torch.Tensor: The computed expected matrix. Shape [B, C, N, N].
    """

    pred = torch.ones_like(snippet[:, 0], dtype=torch.float32).unsqueeze(1)  # [B,1, N, N]
    mask = weight == 0
    weight[mask] = 1
    weight = 1 / weight
    weight[mask] = 0

    # Expand dimensions for broadcasting over N, N
    pred = pred * weight.unsqueeze(3) * weight.unsqueeze(2)  # [B, C, N, N]

    ar = torch.arange(snippet.shape[2], device=snippet.device)  # [N]
    dist = torch.abs(ar.unsqueeze(0) - ar.unsqueeze(1))  # [N, N]

    # Expand and repeat distance matrix to match batch and channel dimensions
    dist_expanded = dist.unsqueeze(0).unsqueeze(0).expand(snippet.shape[0], snippet.shape[1], -1, -1)  # [B, C, N, N]
    dsize = dist_expanded.size()
    dist_expanded = dist_expanded.view((dsize[0], dsize[1], -1))  # [B, C, N*N] because gather works over 1 dim only
    expmat = torch.gather(exp, 2, dist_expanded)  # [B, C, N*N]
    pred = pred * expmat.view(dsize)  # [B, C, N, N]

    final_mask = pred == 0
    snippet[final_mask] = 0
    return snippet, pred  # [B, C, N, N], [B, C, N, N]


def hic_hierarchical_loss(
    pred_ooe: torch.Tensor,  # [B, C, N, N]
    raw: torch.Tensor,  # [B, C, N, N]
    exp_mat: torch.Tensor,  # [B, C, N, N]
    loss_scale_factor: float = 0.5,
    sum_loss_weight: float = 0.2,
    continuity_loss_weight: float = 0.02,
    hierarchical_levels: int = 4,
) -> torch.Tensor:
    """
    Computes the hierarchical loss for Hi-C matrices.

    This function calculates a loss for Hi-C matrices, which combines:
    - Continuity loss to encourage smoothness in predictions over missing bins.
    - Multinomial loss to measure discrepancies between predicted and observed data distributions.
    - Hierarchical loss through recursive aggregation in 2x2 blocks.
    - Sum matching loss to align total sums of predicted and observed matrices.

    Parameters
    ----------
    pred_ooe : torch.Tensor
        Predicted observed-over-expected (OOE) values. Shape [B, C, N, N].
    raw : torch.Tensor
        Ground truth Hi-C contact counts (observed counts). Shape [B, C, N, N].
    exp_mat : torch.Tensor
        Expected matrix for the same region. Shape [B, C, N, N].
    loss_scale_factor : float, optional
        Scale factor for subsequent hierarchical levels. Default is 0.5.
    sum_loss_weight : float, optional
        Weight for the sum matching penalty. Default is 0.2.
    continuity_loss_weight : float, optional
        Weight for the continuity loss. Default is 0.02.
    hierarchical_levels : int, optional
        Number of hierarchical levels to compute. Default is 4.

    Returns
    -------
    torch.Tensor
        Scalar tensor representing the computed hierarchical loss.
    """

    # Constants
    epsilon = 1e-6  # Small value to avoid division by zero and log(0)
    huber_delta = 0.3  # Delta parameter for Huber loss

    # Reshape tensors from [B, C, N, N] to [B*C, N, N]
    B, C, N, _ = raw.shape
    raw = raw.reshape(-1, N, N).detach()  # [B*C, N, N]
    pred_ooe = pred_ooe.reshape(-1, N, N)  # [B*C, N, N]
    exp_mat = exp_mat.reshape(-1, N, N).detach()  # [B*C, N, N]
    batch_size = raw.shape[0]  # B*C

    # Create a mask for positions with zero expected values (those are missing bins)
    mask = exp_mat < epsilon  # [B*C, N, N]

    # Compute continuity loss over gaps
    pred_diff = torch.diff(pred_ooe, dim=1)  # [B*C, N-1, N]
    # [B*C, N-1, N]
    huber_diff_loss = F.huber_loss(pred_diff, torch.zeros_like(pred_diff), delta=huber_delta, reduction="none")

    # We want to penalize only the gaps where one or both positions are missing
    mask1 = mask[:, 1:, :]  # [B*C, N-1, N]
    mask2 = mask[:, :-1, :]  # [B*C, N-1, N]
    combined_mask = mask1 | mask2  # [B*C, N-1, N]
    diff_loss = (huber_diff_loss * combined_mask).sum() / combined_mask.sum()

    # Avoid zeros by adding epsilon
    raw += epsilon
    pred_ooe += epsilon
    exp_mat += epsilon

    # Compute predicted matrix
    pred = pred_ooe * exp_mat  # [B*C, N, N]

    # Zero out masked positions
    pred_ooe[mask] = 0
    raw[mask] = 0
    pred[mask] = 0

    # Normalize each sample to sum to 1
    raw_sums = raw.sum(dim=(1, 2))  # [B*C]
    pred_sums = pred.sum(dim=(1, 2))  # [B*C]
    raw = raw / raw_sums[:, None, None]
    pred = pred / pred_sums[:, None, None]

    # Compute multinomial loss
    valid_positions = raw > 0  # [B*C, N, N]
    multinomial_loss = -(raw[valid_positions] * torch.log(pred[valid_positions])).sum() / batch_size
    scales = 1

    # Hierarchical loss computation
    for level in range(hierarchical_levels):
        # Reshape and sum over 2x2 blocks to downsample
        N = raw.shape[1]
        raw = raw.reshape(raw.shape[0], N // 2, 2, N // 2, 2).sum(dim=(2, 4))  # [B*C, N//2, N//2]
        pred = pred.reshape(pred.shape[0], N // 2, 2, N // 2, 2).sum(dim=(2, 4))  # [B*C, N//2, N//2]

        valid_positions = raw > 0
        scale_factor = loss_scale_factor ** (level + 1)
        multinomial_loss += -(raw[valid_positions] * torch.log(pred[valid_positions])).sum() / batch_size * scale_factor
        scales += 1 * scale_factor

    loss = multinomial_loss / scales
    loss += sum_loss_weight * ((torch.log(pred_sums + 1) - torch.log(raw_sums + 1)) ** 2).mean()
    loss += continuity_loss_weight * diff_loss
    return loss


@torch.no_grad()
def hic_corrs(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes Spearman rank correlation, Pearson correlation, and mean squared difference (MSD)
    between corresponding Hi-C matrices.

    This function calculates the correlations and MSD between corresponding matrices in two tensors,
    excluding any positions where either matrix has zero or non-finite values.

    Parameters
    ----------
    mat1 : torch.Tensor
        First tensor of Hi-C matrices. Shape [B, C, N, N].
    mat2 : torch.Tensor
        Second tensor of Hi-C matrices. Shape [B, C, N, N].

    Returns
    -------
    (spearman_corr, pearson_corr, msd) : tuple of torch.Tensor
        Each of shape [B, C], containing:
          spearman_corr: Spearman rank correlations
          pearson_corr: Pearson correlations
          msd: mean squared differences
    """
    B, C, N, _ = mat1.shape
    mat1 = mat1.reshape(B * C, N * N)
    mat2 = mat2.reshape(B * C, N * N)

    # Create mask of valid positions where both matrices have non-zero, finite values
    valid_mask = (mat1 != 0) & (mat2 != 0) & torch.isfinite(mat1) & torch.isfinite(mat2)

    spearman_corr = torch.empty(B * C, device=mat1.device, dtype=torch.float32)
    pearson_corr = torch.empty(B * C, device=mat1.device, dtype=torch.float32)
    msd = torch.empty(B * C, device=mat1.device, dtype=torch.float32)

    for i in range(B * C):
        x = mat1[i][valid_mask[i]]
        y = mat2[i][valid_mask[i]]

        if x.numel() == 0:
            spearman_corr[i] = float("nan")
            pearson_corr[i] = float("nan")
            msd[i] = float("nan")
            continue

        # Ranks for Spearman (approximate, ties are not handled)
        rx = torch.zeros_like(x).float()
        ry = torch.zeros_like(y).float()
        rx[torch.sort(x)[1]] = torch.arange(len(x), dtype=torch.float32, device=mat1.device)
        ry[torch.sort(y)[1]] = torch.arange(len(y), dtype=torch.float32, device=mat1.device)

        # Spearman via Pearson on ranks
        cov_s = ((rx - rx.mean()) * (ry - ry.mean())).mean()
        spearman_corr[i] = cov_s / (rx.std() * ry.std() + 1e-8)

        # Pearson
        cov_p = ((x - x.mean()) * (y - y.mean())).mean()
        pearson_corr[i] = cov_p / (x.std() * y.std() + 1e-8)

        # Mean squared difference
        msd[i] = (x / y).log().square().mean()

    return (
        spearman_corr.view(B, C),
        pearson_corr.view(B, C),
        msd.view(B, C),
    )


@torch.no_grad()
def coarsegrained_hic_corrs(
    pred_ooe: torch.Tensor,
    raw: torch.Tensor,
    weight: torch.Tensor,
    exp: torch.Tensor,
    cutoff: int = 10,
    also_divide_by_mean: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes Spearman rank correlations between predicted OOE and adaptively coarse-grained OOE from raw data.

    This function calculates the expected matrix from weights and expected values, divides the raw counts by the
    expected matrix to obtain observed-over-expected (OOE) values, applies adaptive coarsegraining to the OOE values
    using the raw counts, and then computes the Spearman rank correlation between the predicted OOE and the adaptively
    coarse-grained OOE.

    Parameters
    ----------
    pred_ooe : torch.Tensor
        Predicted observed-over-expected (OOE) values from the model. Shape [B, C, N, N].
    raw : torch.Tensor
        Raw Hi-C contact counts (observed counts). Shape [B, C, N, N].
    weight : torch.Tensor
        Weight tensor for each channel. Shape [B, C, N].
    exp : torch.Tensor
        Expected value tensor. Shape [B, C, N * 5 / 4].
    cutoff : int, optional
        Minimum number of raw counts per pixel required to stop 2x2 pooling in adaptive coarsegraining. Default is 5.
    also_divide_by_mean : bool, optional
        Whether to also divide the matrices by their mean over channels (but within batch). Default is False.

    Returns
    -------
    If also_divide_by_mean is False:
        (spearman_corr, pearson_corr, msd) : tuple of torch.Tensor
    If also_divide_by_mean is True:
        (spearman_corr, pearson_corr, msd, spearman_corr2, pearson_corr2, msd2) : tuple of torch.Tensor
        Each of shape [B, C], containing:
          spearman_corr: Spearman rank correlations
          pearson_corr: Pearson correlations
          msd: mean squared differences
          <something>2: Corresponding values after dividing by mean over channels
    """

    # Calculate expected matrix from weights and expected values
    raw, expected_matrix = create_expected_matrix(raw.clone(), weight, exp)

    # Compute OOE (observed-over-expected)
    # Avoid division by zero by setting expected_matrix zeros to NaN
    exp_zero_mask = expected_matrix == 0
    expected_matrix[exp_zero_mask] = 1.0
    ooe = raw / expected_matrix
    # Set positions where expected_matrix was zero to zero in ooe
    ooe[exp_zero_mask] = 0

    # Apply adaptive coarsegraining to OOE values using raw counts
    adaptive_smoothed_ooe = adaptive_coarsegrain_torch(ooe, raw, cutoff=cutoff)

    # Compute Spearman correlation between predicted OOE and adaptively coarse-grained OOE
    corrs = hic_corrs(pred_ooe, adaptive_smoothed_ooe)

    if also_divide_by_mean:
        # Divide by mean over channels (but within batch) from both matrices
        pred_ooe = pred_ooe / pred_ooe.mean(dim=1, keepdim=True)
        adaptive_smoothed_ooe = adaptive_smoothed_ooe / adaptive_smoothed_ooe.mean(dim=1, keepdim=True)
        corrs2 = hic_corrs(pred_ooe, adaptive_smoothed_ooe)
        corrs = list(corrs) + list(corrs2)

    return tuple(corrs)
