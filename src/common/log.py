"""Tiny logging helper: prints to the notebook / console and (optionally) to a log file."""
import logging


class _PrintHandler(logging.Handler):
    def emit(self, record):
        print(self.format(record))


def get_logger(stage: str) -> logging.Logger:
    log = logging.getLogger(f"recsys.{stage}")
    log.setLevel(logging.INFO)
    root = logging.getLogger("recsys")
    root.setLevel(logging.INFO)
    if not any(isinstance(h, _PrintHandler) for h in root.handlers):
        h = _PrintHandler()
        h.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
        root.addHandler(h)
    return log
