import pandas as pd

from src.crawler.dataupdater import DataUpdater
from src.config.model_config import EXCHANGE_RATE_LIST

if __name__ == "__main__" :
    updater = DataUpdater(EXCHANGE_RATE_LIST)

    updater.update(
    ).save(
    )