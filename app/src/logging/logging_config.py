import logging
import os

def setup_logging(
    log_level="INFO",
    log_dir="logs",
    log_file="app.log"
):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    
    if os.path.exists(log_path):
        print(f"Remove {log_path}")
        os.remove(log_path)
        
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    file_handler = logging.handlers.RotatingFileHandler(
        log_path, 
        mode='w', 
        maxBytes=1_000_000, 
        backupCount=3,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)
    
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)