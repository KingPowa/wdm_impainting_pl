import os
import logging
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import DictConfig

from ml.core.utils.etc import get_timestamp

class Session:

    def __init__(self, name: str, config: DictConfig):
        self.session_identifier = get_timestamp()
        self.working_directory = os.path.join(config.train_config.base_dir, f"{name}_exec_{self.session_identifier}")
        self.config = config
        self.name = name

        os.makedirs(self.working_directory, exist_ok=True)

    @property
    def log_dir(self):
        log_dir = os.path.join(self.working_directory, self.__get_log_dir())
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
    
    def __get_log_dir(self):
        return (self.config.train_config.log_dir if "log_dir" in self.config.train_config else "logs")
    
    @property
    def ckpt_dir(self):
        ckdir = os.path.join(self.working_directory, self.__get_ckpt_dir())
        os.makedirs(ckdir, exist_ok=True)
        return ckdir
    
    def __get_ckpt_dir(self):
        return (self.config.train_config.chkpt_dir if "chkpt_dir" in self.config.train_config else "checkpoints")
    
    @property
    def devices(self):
        return self.config.train_config.devices if "devices" in self.config.train_config else 1
    
    @property
    def accelerator(self):
        return self.config.train_config.accelerator if "accelerator" in self.config.train_config else "auto"
    
    def is_slurm(self):
        return 'SLURM_JOB_ID' in os.environ

def setup_logger(session: Session, level=logging.INFO):
    """Set up a basic logger with timestamp."""
    logger = logging.getLogger(f"{session.name}_Logger")
    logger.setLevel(level)

    # Create a file handler to log to a file
    file_handler = logging.FileHandler(os.path.join(session.log_dir, f"activity_{session.session_identifier}.log"))
    file_handler.setLevel(level)

    # Define the log format with timestamp
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

    return logger

class AdvancedWandLogger(WandbLogger):

    def __init__(self, 
                 model,
                 session: Session,
                 project: str = None,
                 version = None, 
                 offline = False,
                 **kwargs):
        
        name = f"{model._get_name()}#{session.session_identifier}"
        save_dir = session.log_dir
        if project is None:
            project = f"{model._get_name()}_project"
        
        super().__init__(name, save_dir, version, offline, None, None, None, project, "all", None, None, None, **kwargs)

class AdvancedModelCheckpoint(ModelCheckpoint):

    def __init__(self,
                 session: Session,
                 monitor: str,
                 mode: str = "min",
                 filename_suffix: str = "",
                 save_last = None, 
                 save_top_k = 3, 
                 every_n_train_steps = None, 
                 train_time_interval = None, 
                 every_n_epochs = None, 
                 save_on_train_epoch_end = None, 
                 enable_version_counter = True):
        
        dirpath = session.ckpt_dir
        filename = f"{filename_suffix}_" + "{epoch:02d}-{val_loss:.4f}"
        
        super().__init__(dirpath, 
                         filename, 
                         monitor, 
                         False, 
                         save_last, 
                         save_top_k, 
                         False, 
                         mode, 
                         True, 
                         every_n_train_steps, 
                         train_time_interval, 
                         every_n_epochs, 
                         save_on_train_epoch_end, 
                         enable_version_counter)