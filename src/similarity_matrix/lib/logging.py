
import logging

DEFAULT_LOG_LEVEL = logging.INFO

# Create a named logger
logger = logging.getLogger('similarity_matrix')

# Set the logging level
logger.setLevel(DEFAULT_LOG_LEVEL)

# Create a console handler (or FileHandler, etc.)
console_handler = logging.StreamHandler()

# Set level for the handler (optional)
console_handler.setLevel(DEFAULT_LOG_LEVEL)

# Create and set a formatter
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# console_handler.setFormatter(formatter)

# Add handler to the logger
logger.addHandler(console_handler)
