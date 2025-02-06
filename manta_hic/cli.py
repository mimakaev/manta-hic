# manta_hic/cli.py

import click

from .io.cool_io import process_mcools as process_mcools_original
from .nn.manta import populate_microzoi_cache as fill_cache_original
from .nn.mutate_manta import manta_mutate_file
from .nn.train_manta import train_manta_click
from .nn.train_microzoi import train_microzoi


@click.command(context_settings={"show_default": True})
@click.option("--manifest-path", "-m", type=click.Path(exists=True), required=True, help="Path to the manifest CSV.")
@click.option(
    "--output-folder",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True),
    required=True,
    help="Folder where the outputs (HDF5 files) are written.",
)
@click.option("--resolutions", "-r", default="1024,2048,4096,8192", help="Comma-separated list of resolutions.")
@click.option("--target-size", default=1024, help="Window size needed for the neural network, in bins.")
@click.option("--step-bins", default=256, help="Step size in bins.")
def process_mcools(manifest_path, output_folder, resolutions, target_size, step_bins):
    """
    Read a manifest CSV that has columns: group_name, filepath, genome.
    Group by group_name, produce one HDF5 file per resolution per group.
    """
    process_mcools_original(manifest_path, output_folder, resolutions, target_size, step_bins)


@click.command(context_settings={"show_default": True})
@click.option("--cache-path", "-c", type=click.Path(), required=True, help="Path to the cachefile.")
@click.option("--modfile", "-m", type=click.Path(exists=True), required=True, help="Path to the mod file.")
@click.option("--fasta", "-f", type=click.Path(exists=True), required=True, help="Path to the FASTA file.")
@click.option("--device", "-d", default="cuda:0", help="Torch device")
@click.option("--batch-size", "-b", default=4, help="Batch size.")
@click.option("--chrom", type=str, multiple=True, default=["#", "chrX"], help="Chromosomes to process.")
@click.option("--params-file", "-p", type=click.Path(exists=True), help="Path to the parameters file.")
@click.option("--n-runs", "-n", default=16, help="Number of runs.")
@click.option("--crop-mha-range", "-r", type=(int, int), default=(640, 1024), help="Crop MHA range.")
@click.option("--max-shift-bp", default=128, help="Maximum shift in base pairs.")
@click.option("--n-channels", default=1024 + 8, help="Number of channels in the output of model.")
def fill_cache(
    cache_path,
    modfile,
    fasta,
    chrom=("#", "chrX"),
    params_file=None,
    n_runs=16,
    crop_mha_range=(640, 1024),
    max_shift_bp=128,
    batch_size=4,
    n_channels=1024 + 8,
    device="cuda:0",
):
    fill_cache_original(
        cache_path,
        modfile=modfile,
        fasta=fasta,
        chroms=chrom,
        params_file=params_file,
        N_runs=n_runs,
        crop_mha_range=crop_mha_range,
        max_shift_bp=max_shift_bp,
        batch_size=batch_size,
        n_channels=n_channels,
        device=device,
    )


@click.group()
def cli():
    """Manta HIC Command Line Interface."""
    pass


@cli.group()
def train():
    """Training related commands."""
    pass


train.add_command(train_microzoi)
train.add_command(train_manta_click)


@cli.group()
def io():
    """Input/Output related commands."""
    pass


io.add_command(process_mcools)
io.add_command(fill_cache)


@cli.group()
def mutate():
    """Mutation screens."""
    pass


mutate.add_command(manta_mutate_file)

if __name__ == "__main__":
    cli()
