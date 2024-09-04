import ray

from dotenv import load_dotenv
load_dotenv()

from src.crawler import RealTimeExRateCrawler
from src.config import CRAWLER_CONFIG
from src.config import EXCHANGE_TYPES
from src.config import PROCESS_BOUNDARY

if __name__ == "__main__" :
    # crawlers = [RealTimeExRateCrawler.remote(
    #                 CRAWLER_CONFIG['backend_blue_1'], EXCHANGE_TYPES, 
    #                 0, 58
    #             ),
    #             RealTimeExRateCrawler.remote(
    #                 CRAWLER_CONFIG['backend_blue_2'], EXCHANGE_TYPES, 
    #                 0, 58
    #             )]
    crawlers = [RealTimeExRateCrawler.remote(
                    CRAWLER_CONFIG['backend_green_1'], EXCHANGE_TYPES, 
                    0, 58
                ),
                RealTimeExRateCrawler.remote(
                    CRAWLER_CONFIG['backend_green_2'], EXCHANGE_TYPES, 
                    0, 58
                )]
    _ = ray.get([crawler.run.remote() for crawler in crawlers])