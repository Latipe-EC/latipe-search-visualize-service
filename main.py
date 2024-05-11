import os
import tensorflow as tf

os.environ['CUDA_DIR'] = '/opt/cuda'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/opt/cuda'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import threading
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

from search_api import app  # Import the Flask application
import rabbitmq_consumer  # Import the module that sets up the RabbitMQ consumer

# Read the PORT variable from the environment
port = os.getenv('FLASK_APP_PORT')


def run_flask():
    """Function to run the Flask application."""
    # Ensure the Flask app runs on all interfaces and uses the correct port
    app.run(host='0.0.0.0', port=port, debug=True)


def run_rabbitmq():
    """Function to run the RabbitMQ consumer."""
    rabbitmq_consumer.setup_rabbitmq_consumer()


if __name__ == '__main__':
    # Create a thread for RabbitMQ consumer
    print("Starting RabbitMQ thread...")
    rabbitmq_thread = threading.Thread(target=run_rabbitmq)

    # Start RabbitMQ thread
    rabbitmq_thread.start()
    print("RabbitMQ thread started")

    # Run the Flask application in the main thread
    print("Starting Flask application...")
    run_flask()

    # Join RabbitMQ thread to the main thread to keep it running
    rabbitmq_thread.join()
