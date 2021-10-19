import logging as log


class Logger:
    log.basicConfig(level=log.INFO, filename="logger.log", format="[%(asctime)s] %(levelname)s: %(message)s",
                        datefmt="%d/%m/%Y %H:%M:%S")

    def basicConfig(*args, **kwargs):
        log.basicConfig(*args, **kwargs)

    def info(*args, **kwargs):
        log.info(*args, **kwargs)

    def debug(*args, **kwargs):
        log.debug(*args, **kwargs)

    def error(*args, **kwargs):
        log.error(*args, **kwargs)

    def warn(*args, **kwargs):
        log.warn(*args, **kwargs)
