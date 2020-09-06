import logging


class AppLogger(logging.Logger):
    _bold_seq = "\033[1m"
    _fmt_reset = "\033[0m"
    default_log_level = logging.DEBUG
    log_format = f"{_bold_seq}[%(asctime)s | %(funcName)s - %(levelname)s]{_fmt_reset} %(message)s"
    date_format = "%H:%M:%S"

    def __init__(self, name: str, log_level=default_log_level):
        logging.setLoggerClass(type(self))
        super().__init__(name)
        formatter = logging.Formatter(fmt=self.log_format, datefmt=self.date_format)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.addHandler(handler)
        self.setLevel(log_level)
