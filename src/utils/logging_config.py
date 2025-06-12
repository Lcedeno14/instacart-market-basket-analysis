import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logging(module_name: str = None, log_level: str = 'INFO') -> logging.Logger:
    """
    Set up logging configuration for a module.
    
    Args:
        module_name: Name of the module (e.g., 'app', 'api', 'etl')
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Determine log file name
    if module_name:
        log_file = logs_dir / f"{module_name}.log"
    else:
        log_file = logs_dir / "general.log"
    
    # Convert string log level to logging constant
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    log_level = level_map.get(log_level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger(module_name or __name__)
    logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_logger(module_name: str = None) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        module_name: Name of the module
    
    Returns:
        Logger instance
    """
    return logging.getLogger(module_name or __name__)

# Global logging configuration
def setup_global_logging():
    """Set up global logging configuration for the application."""
    # Create logs directory
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler for general logs
    general_log_file = logs_dir / "general.log"
    file_handler = logging.FileHandler(general_log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Set specific log levels for modules
    logging.getLogger('src.analysis.data_processor').setLevel(logging.INFO)
    
    return root_logger 