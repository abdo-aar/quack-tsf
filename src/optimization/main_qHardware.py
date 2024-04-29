import os.path
import omegaconf
from qiskit_ibm_runtime import QiskitRuntimeService, Session

from src.optimization.bo_process import main_wrapper
from src.utils.settings import PROJECT_ROOT_PATH

if __name__ == "__main__":
    # Upload the config manually
    config_session = os.path.join(PROJECT_ROOT_PATH, 'src', 'optimization', 'configs', 'config_session.yaml')
    cfg = omegaconf.OmegaConf.load(config_session)

    # This should be previously configured to the right user token
    service = QiskitRuntimeService(channel=cfg.channel, token=cfg.token, instance=cfg.instance)

    # # Choose the configuration file based on whether it's a multi-run
    config_name = "config_run"  # Only single run experiments are allowed when working with IBM_Q

    with Session(service=service, backend=cfg.backend):
        main_wrapper(config_name=config_name, instance=cfg.instance, backend=cfg.backend)
