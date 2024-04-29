import sys

from src.optimization.bo_process import main_wrapper

if __name__ == "__main__":
    # Detect if running a multi-run based on command-line arguments
    is_multi_run = '-m' in sys.argv or '--multirun' in sys.argv

    # Choose the configuration file based on whether it's a multi-run
    config_name = "config_multirun" if is_multi_run else "config_run"

    main_wrapper(config_name)
