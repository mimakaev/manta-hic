import json
import os
import pickle
import random
import shutil
from contextlib import contextmanager, nullcontext

import click
import numpy as np
import torch
import torch.optim as optim

from manta_hic.nn.manta import (
    HiCDataset,
    HybridCachedStochasticFetcher,
    Manta2,
    ThreadedDataLoader,
    run_epoch,
)


@contextmanager
def ephemeral_copy(src_path: str, dst_path: str):
    """
    Copies src_path to dst_path. Yields dst_path, then automatically
    removes it on exit or exception.
    """
    print(f"Copying {src_path} to {dst_path}")
    shutil.copy2(src_path, dst_path)
    print(f"Copied {src_path} to {dst_path}")
    try:
        yield dst_path
    finally:
        if os.path.exists(dst_path):
            os.remove(dst_path)


file = click.Path(exists=True, dir_okay=False)


@click.command(name="manta", context_settings={"show_default": True})
@click.option("--input-file", "-i", type=file, required=True, help="Path to the datafile.")
@click.option("--cache-path", "-c", type=file, required=True, help="Path to the cachefile, should be on SSD")
@click.option("--output-folder", "-o", type=click.Path(file_okay=False), required=True, help="Output folder.")
@click.option("--fasta-path", "-f", type=file, required=True, help="Path to the FASTA file.")
@click.option("--device", "-d", default="cuda:0", help="Torch device")
@click.option("--genome", "-g", default="hg38", help="Genome")
@click.option("--n-epochs", "-e", default=0, help="Number of epochs. (0 is auto)")
@click.option("--work-dir", default=None, help="Working directory in a fast location to store the datafile.")
@click.option("--overwrite", is_flag=True, help="Overwrite the output folder.")
@click.option("--params", type=click.Path(exists=True), help="Path to the parameters file JSON file")
@click.option("--batch-size", default=2, help="Batch size.")
@click.option("--lr", default=0.0001, help="Learning rate.")
@click.option("--save-every", default=10, help="Save model every n epochs.")
@click.option("--n-bins", default=1024, help="Number of bins in the Hi-C map")
@click.option("--bins-pad", default=128, help="Number of padding bins.")
@click.option("--val-fold", default="fold3", help="Validation fold")
@click.option("--test-fold", default="fold4", help="Test fold")
@click.option("--use-all-data", is_flag=True, help="Use all data without train/test splits")
@click.option("--epoch-multiplier", default=1.0, type=float, help="Multiplier for number of epochs")
def train_manta_click(
    input_file,
    cache_path,
    output_folder,
    fasta_path,
    device="cuda:0",
    genome="hg38",
    params=None,
    overwrite=False,
    batch_size=2,
    n_epochs=0,
    work_dir=None,
    lr=0.0001,
    save_every=10,
    n_bins=1024,
    bins_pad=128,
    val_fold="fold3",
    test_fold="fold4",
    use_all_data=False,
    epoch_multiplier=1.0,
):
    if os.path.exists(output_folder):
        if overwrite:
            shutil.rmtree(output_folder)
        else:
            raise ValueError(f"Output folder {output_folder} already exists.")
    os.makedirs(output_folder, exist_ok=True)

    if params is not None:
        params = json.load(open(params))

    if work_dir is not None:  # need to copy the datafile to the work_dir in a safe way
        random_suffix = str(random.randint(0, 1000000)) + "_"
        copied_input_file = os.path.join(work_dir, random_suffix + os.path.basename(input_file))
        manager = ephemeral_copy(input_file, copied_input_file)
    else:
        manager = nullcontext(input_file)
    with manager as input_file:
        train_manta(
            input_file,
            cache_path,
            output_folder,
            fasta_path,
            device,
            genome,
            params,
            batch_size,
            n_epochs,
            lr,
            save_every,
            n_bins,
            bins_pad,
            test_fold,
            val_fold,
            use_all_data,
            epoch_multiplier,
        )


def train_manta(
    input_file,
    cache_path,
    output_folder,
    fasta_path,
    device="cuda:0",
    genome="hg38",
    params=None,
    batch_size=2,
    n_epochs=0,
    lr=0.0001,
    save_every=10,
    n_bins=1024,
    bins_pad=128,
    val_fold="fold3",
    test_fold="fold4",
    use_all_data=False,
    epoch_multiplier=1.0,
):
    if params is None:
        params = {}

    fetcher = HybridCachedStochasticFetcher(cache_path, fasta_path, prob_mean=0.1, max_mean_runs=6)

    if use_all_data:
        ds_train = HiCDataset(
            input_file,
            fetcher,
            n_bins=n_bins,
            bins_pad=bins_pad,
            genome=genome,
            fold_types_use=["all"],
            stochastic_offset=True,
            stochastic_reverse=True,
        )

    else:
        ds_train = HiCDataset(
            input_file,
            fetcher,
            n_bins=n_bins,
            bins_pad=bins_pad,
            genome=genome,
            test_fold=test_fold,
            val_fold=val_fold,
            stochastic_offset=True,
            stochastic_reverse=True,
        )

        ds_val = HiCDataset(
            input_file,
            fetcher,
            n_bins=n_bins,
            bins_pad=bins_pad,
            genome=genome,
            fold_types_use=["val"],
            test_fold=test_fold,
            val_fold=val_fold,
            stochastic_offset=False,
            stochastic_reverse=False,
        )

    assert len(ds_train) > 0, "No training data"
    assert len(ds_val) > 0, "No validation data"

    train_dl = ThreadedDataLoader(ds_train, batch_size=batch_size, shuffle=True, fraction=0.5)
    val_dl = ThreadedDataLoader(ds_val, batch_size=batch_size * 2, shuffle=False, fraction=1)

    hic_res = ds_train.hic_res
    # resolution of microzoi is 256bp, which has log2(256)=8. plus one because we maxpool after the first convolution
    # which is not in a tower. Also tower is split to be before/after MHA, with one layer after MHA, so minimum
    # resolution of the model is 1024bp:
    # microzoi (256bp) -> maxpool (512bp) -> MHA (512bp) -> last conv block + maxpool (1024bp) -> Hi-C map (1024bp))
    params["tower_height"] = int(np.round(np.log2(hic_res))) - 9

    res_epoch_dict = {256: 20, 512: 30, 1024: 40, 2048: 50, 4096: 60, 8192: 70, 16384: 80}
    if n_epochs == 0:
        n_epochs = res_epoch_dict[hic_res]
    n_epochs = int(n_epochs * epoch_multiplier)

    model = Manta2(**params, output_channels=ds_train.n_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.GradScaler()
    os.makedirs(output_folder, exist_ok=True)

    for epoch in range(n_epochs):
        model.train()
        corrs_train = run_epoch(
            model,
            train_dl,
            device=device,
            is_train=True,
            optimizer=optimizer,
            scaler=scaler,
        )

        # -- Validate
        if not use_all_data:
            corrs_val = run_epoch(model, val_dl, device=device, is_train=False)

        if (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), f"{output_folder}/model_{epoch}.pth")

        with open(f"{output_folder}/corrs_{epoch}.pkl", "wb") as f:
            pickle.dump([corrs_train, corrs_val], f)

        ct = ", ".join([f"{i:.3f}" for i in np.mean(corrs_train, axis=(0, 2, 3))])
        cv = ", ".join([f"{i:.3f}" for i in np.mean(corrs_val, axis=(0, 2, 3))])

        print(f"Epoch {epoch+1}/{n_epochs}: train_corrs={ct}, val_corr={cv}   ")
    torch.save(model.state_dict(), os.path.join(output_folder, "saved_model.pth"))
