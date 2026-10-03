"""Application-owned standard-library logging configuration."""

import logging.config


def configure_logging() -> None:
    """Configure Code Partner logs for the temporary protocol investigation."""
    # ##DELETE AFTER CORRECTION## Restore the codepartner logger to INFO once the
    # OpenAI/PyCharm tool-response investigation is complete.
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",  # ##DELETE AFTER CORRECTION##
                "formatter": "standard",
            },
        },
        "loggers": {
            "codepartner": {
                "handlers": ["console"],
                "level": "DEBUG",  # ##DELETE AFTER CORRECTION##
                "propagate": False,
            },
        },
        "root": {"handlers": ["console"], "level": "INFO"},
    })
