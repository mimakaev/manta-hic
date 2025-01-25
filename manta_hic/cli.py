# manta_hic/cli.py

import click

from .nn.train_microzoi import train_microzoi
from .io.cool_io import process_mcools as process_mcools_original


@click.command()
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


@click.group()
def cli():
    """Manta HIC Command Line Interface."""
    pass


@cli.group()
def train():
    """Training related commands."""
    pass


train.add_command(train_microzoi)


@cli.group()
def io():
    """Input/Output related commands."""
    pass


io.add_command(process_mcools)

if __name__ == "__main__":
    cli()
