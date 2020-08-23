import logging


class Logger(logging.Logger):
    _bold_seq = "\033[1m"
    _fmt_reset = "\033[0m"
    log_format = f"{_bold_seq}[%(asctime)s | %(funcName)s - %(levelname)s]{_fmt_reset} %(message)s"
    date_format = "%H:%M:%S"

    def __init__(self, name: str):
        logging.setLoggerClass(type(self))
        super().__init__(name)
        formatter = logging.Formatter(fmt=self.log_format, datefmt=self.date_format)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.addHandler(handler)
