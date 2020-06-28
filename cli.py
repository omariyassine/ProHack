""" 
Command Line Instructions (CLI) to excute the scripts in computing 
"""

import click
import logging

from core.logging import init_logger
from utils import (
    log_title,
    drop_if_in_data,
    concat_providers_data,
)

init_logger()
logger = logging.getLogger(__name__)


@click.group()
def main():
    pass


@main.command()
@click.option(
    "--debug", help="Run scraper using browser graphical interface", is_flag=True,
)
def run_regression(debug: bool) -> None:
    """Run Regression part
    """


def run_optimisation(debug: bool) -> None:
    """Run Regression part
    """


if __name__ == "__main__":
    main()
