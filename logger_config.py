# logger_config.py

import logging
import os
import sys

import coloredlogs

# Configure the logger
log_format = '\n%(asctime)s - %(levelname)s - %(filename)s -- %(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(f'log/{os.path.basename(__file__)}.log'),
        logging.StreamHandler(sys.stdout),
    ])

# Define a custom color scheme
field_styles = {
    'asctime': {'color': 'green'},
    'levelname': {'color': 'black', 'bold': True},
    'filename': {'color': 'magenta'}  # Set filename field to blue
}
level_styles = {
    'debug': {'color': 'blue'},
    'info': {'color': 'black'},
    'warning': {'color': 'yellow'},
    'error': {'color': 'red'},
    'critical': {'color': 'red', 'bold': True}
}

# Apply coloredlogs with the custom styles
coloredlogs.install(level='INFO', logger=logging.getLogger(__name__), fmt=log_format, level_styles=level_styles,
                    field_styles=field_styles)

# Create and export a logger instance
logger = logging.getLogger()
