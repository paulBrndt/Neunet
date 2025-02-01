import logging

def setup():
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-&d %H:%M")

setup()
logging.debug("Hi there")

class Logger:

    def __init__(self, filename: str) -> None:
        pass