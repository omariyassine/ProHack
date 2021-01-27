""" Utils Functions

"""
import pandas as pd
import time
import logging


import config.own as config


logger = logging.getLogger(__file__)


def log_title(title, lenght=100):
    title = (
        "\n"
        + lenght * "-"
        + "\n"
        + ((lenght // 2 - 1) - len(title) // 2) * "-"
        + f" {title} "
        + ((lenght // 2 - 1) - len(title) // 2 - len(title) % 2) * "-"
        + "\n"
        + lenght * "-"
        + "\n"
    )
    logging.info(title)


def drop_if_in_data(data, *fields):
    to_drop = []
    for field in fields:
        if field in data.columns:
            to_drop.append(field)

    data = data.drop(to_drop, axis=1)

    return data


def concat_data_by_path(*paths):
    to_concat = []
    for path in paths:
        to_concat.append(pd.read_pickle(path))
    providers = pd.concat(to_concat, axis=0)
    providers = drop_if_in_data(providers, "PROVIDER_ID")
    providers["nan_percentage"] = providers.isna().mean(axis=1)

    providers = (
        providers.sort_values(by=["nan_percentage"])
        .drop_duplicates(subset=[config.PROVIDER_NAME_COLUMN])
        .drop("nan_percentage", axis=1)
    )

    providers = providers.reset_index(drop=True)
    providers.index.names = ["PROVIDER_ID"]
    providers = providers.reset_index()

    providers.to_pickle(config.GOOGLE_API_INPUT)


def _find_element_click(
    driver,
    by,
    expression,
):
    """Find the element and click then  handle all type of exception during click

    Args:
        driver (selenium.driver): Selenium driver
        by (selenium.webdriver.common.by): Type of selector
            (By.XPATH, By.CSSSelector ...)
        expression (str): Selector expression to the element to click on

    Returns:
        bool: True if successfully clicked on the element
    """
    end_time = time.time() + 32
    while True:
        try:
            web_element = driver.find_element(by=by, value=expression)
            web_element.click()
            return True
        except:
            if time.time() > end_time:
                time.sleep(4)
                break
    return False
