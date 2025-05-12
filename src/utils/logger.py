import logging
import sys
import shutil

def setup_logger(log_file=None, backup_file=None):
    logger = logging.getLogger("XlitTask")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    formatter = logging.Formatter("%(asctime)s (%(module)s:%(lineno)d)  %(levelname)s: %(message)s")

    if log_file.exists():
        shutil.copy(log_file, backup_file)
        log_file.unlink()
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.addHandler(fh)
    logger.addHandler(sh)
    
    return logger
