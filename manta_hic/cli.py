# manta_hic/cli.py

import click

from .nn.train_microzoi import train_microzoi


@click.group()
def cli():
    """Manta HIC Command Line Interface."""
    pass


@cli.group()
def train():
    """Training related commands."""
    pass


train.add_command(train_microzoi)

if __name__ == "__main__":
    cli()
