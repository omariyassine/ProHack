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
    "--evaluation",
    help="Run evaluation, print and save scores",
    is_flag=True,
)
@click.option(
    "--regression",
    help="Deploy model ans save output",
    is_flag=True,
)
@click.option(
    "--optimisation",
    help="Deploy model ans save output",
    is_flag=True,
)
def solve_problem(evaluation: bool, regression: bool, optimisation: bool) -> None:
    """Run Regression part"""
    if evaluation:
        log_title("START EVALUATION OF THE MODEL")
        evaluate_model()
        log_title("START EVALUATION OF THE MODEL")

    elif regression:
        log_title("START DEPLOYMENT OF THE MODEL")
        deploy_model()
        log_title("END DEPLOYMENT OF THE MODEL")

    elif optimisation:
        log_title("START OPTIMIZATION")
        get_optimal_ditrib()
        log_title("START OPTIMIZATION")

    else:
        log_title("START EVALUATION OF THE MODEL")
        evaluate_model()
        log_title("END EVALUATION OF THE MODEL")

        log_title("START DEPLOYMENT OF THE MODEL")
        deploy_model()
        log_title("END DEPLOYMENT OF THE MODEL")

        log_title("START OPTIMIZATION")
        get_optimal_ditrib()
        log_title("START OPTIMIZATION")


if __name__ == "__main__":
    main()
