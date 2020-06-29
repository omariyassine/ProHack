""" 
Command Line Instructions (CLI) to excute the scripts in computing 
"""

import click
import logging

from core.logging import init_logger
from utils import log_title
from regression import evaluate_model, deploy_model
from optimization import get_optimal_ditrib

init_logger()
logger = logging.getLogger(__name__)


@click.group()
def main():
    pass


@main.command()
@click.option(
    "--val", help="Run evaluation, print and save scores", is_flag=True,
)
@click.option(
    "--deploy", help="Deploy model ans save output", is_flag=True,
)
def run_regression(val: bool, deploy: bool) -> None:
    """Run Regression part
    """
    if val:
        log_title("START EVALUATION OF THE MODEL")
        evaluate_model()
        log_title("START EVALUATION OF THE MODEL")

    elif deploy:
        log_title("START DEPLOYMENT OF THE MODEL")
        deploy_model()
        log_title("END DEPLOYMENT OF THE MODEL")

    else:
        log_title("START EVALUATION OF THE MODEL")
        evaluate_model()
        log_title("END EVALUATION OF THE MODEL")

        log_title("START DEPLOYMENT OF THE MODEL")
        deploy_model()
        log_title("END DEPLOYMENT OF THE MODEL")


@main.command()
def run_optimisation() -> None:
    """Run Regression part
    """
    log_title("START OPTIMIZATION")
    get_optimal_ditrib()
    log_title("START OPTIMIZATION")


if __name__ == "__main__":
    main()
