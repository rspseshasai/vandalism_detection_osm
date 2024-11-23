import logging
import os
import sys
import coloredlogs

# Configure the logger format
log_format = '\n%(asctime)s - %(levelname)s - %(filename)s -- %(message)s'

# Get the directory where this script (logger_config.py) is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Directory where the log file will be saved (in the same directory as logger_config.py)
log_dir = os.path.join(current_dir, 'log')

# Create the log directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Log file path (in the 'log' directory inside the current directory)
log_file_path = os.path.join(log_dir, f'{os.path.basename(__file__)}.log')

# Configure logging to file and stdout
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(log_file_path),  # Log to a file
        logging.StreamHandler(sys.stdout),  # Log to stdout (console)
    ]
)

# Define a custom color scheme for coloredlogs
field_styles = {
    'asctime': {'color': 'green'},
    'levelname': {'color': 'black', 'bold': True},
    'filename': {'color': 'magenta'}
}
level_styles = {
    'debug': {'color': 'blue'},
    'info': {'color': 'black'},
    'warning': {'color': 'yellow'},
    'error': {'color': 'red'},
    'critical': {'color': 'red', 'bold': True}
}

# Apply coloredlogs with the custom styles
coloredlogs.install(level='INFO', logger=logging.getLogger(__name__), fmt=log_format,
                    level_styles=level_styles, field_styles=field_styles)

# Create and export a logger instance
logger = logging.getLogger()