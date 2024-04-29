import logging
import os
import sys

from hydra.conf import HydraConf
from omegaconf import DictConfig


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


def register_logging(log_file_path: str):
    """
    Redirects outputs of ax and other kind of outputs into the provided log file

    :param log_file_path: path to log file
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logger.handlers = []

    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    logger.addHandler(file_handler)

    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

    # Catch ax's logger
    ax_logger = logging.getLogger('ax')

    # Catch hydra's logger
    hydra_logger = logging.getLogger('hydra')

    # Remove StreamHandler from Ax and Hydra loggers to prevent output to the terminal
    for logg in [ax_logger, hydra_logger]:
        for handler in logg.handlers[:]:  # Iterate over a copy of the list to safely remove items
            if isinstance(handler, logging.StreamHandler):
                logg.removeHandler(handler)

        logg.setLevel(logging.INFO)  # Ensure each logger respects the desired log level
        logg.addHandler(file_handler)  # Use the same file_handler for every logger


def logging_setup(cfg: DictConfig, hydra_cfg: HydraConf, handle_logs: bool = True):
    """
    Sets up logging into hydra created the log file, as well as explicitly changes the cwd into the one we are rerunning

    :param cfg: the config file used
    :param hydra_cfg: the hydra config file
    :param handle_logs: additional control parameter that can be set to False for debugging purposes
    """
    # Change path to the correct working directory when rerunning a Hydra experiment
    if "--experimental-rerun" in sys.argv:
        rerun_index = sys.argv.index("--experimental-rerun")
        if rerun_index + 1 < len(sys.argv):
            experiment_dir_path = os.path.dirname(os.path.dirname(sys.argv[rerun_index + 1]))
            os.chdir(experiment_dir_path)  # Change path explicitly
        else:
            raise Exception("'--experimental-rerun' argument provided but no path specified.")

    if cfg.bo_optim.specs.logs.direct_handling and handle_logs:
        # Set up the logger object after hydra's initialization
        log_filename = f"{hydra_cfg.job.name}.log"

        log_file_path = os.path.join(os.getcwd(), log_filename)
        register_logging(log_file_path)


class SafeFileHandler(logging.FileHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        # Ensure the directory exists before initializing the base FileHandler
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        super().__init__(filename, mode, encoding, delay)
