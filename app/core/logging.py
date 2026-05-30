import logging

import sys

from pathlib import Path

from app.core.config import settings





Path("logs").mkdir(exist_ok=True)





def setup_logging():

    """Configure application logging"""





    formatter = logging.Formatter(

        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",

        datefmt="%Y-%m-%d %H:%M:%S"

    )





    console_handler = logging.StreamHandler(sys.stdout)

    console_handler.setFormatter(formatter)





    file_handler = logging.FileHandler("logs/app.log")

    file_handler.setFormatter(formatter)





    root_logger = logging.getLogger()

    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL))

    root_logger.addHandler(console_handler)

    root_logger.addHandler(file_handler)





    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    logging.getLogger("httpx").setLevel(logging.WARNING)



    return root_logger





logger = setup_logging()
