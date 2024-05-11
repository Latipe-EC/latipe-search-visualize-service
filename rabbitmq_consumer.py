# rabbitmq_consumer.py
import json
import os

import numpy as np
import pika
from dotenv import load_dotenv

import mongodb_handler
from model_utils import extract_features, train_resnet50_model
from keras.preprocessing import image

# Load the .env file
load_dotenv()

# Read the PORT variable from the environment
RABBITMQ_HOST = os.getenv('RABBITMQ_HOST')
AUTH_SERVICE_URL = os.getenv('AUTH_SERVICE_URL')
PRODUCT_UPDATES_QUEUE = os.getenv('PRODUCT_UPDATES_QUEUE')
SCHEDULE_QUEUE = os.getenv('SCHEDULE_QUEUE')
PRODUCT_EXCHANGE = os.getenv('PRODUCT_EXCHANGE')
SCHEDULE_EXCHANGE = os.getenv('SCHEDULE_EXCHANGE')
PRODUCT_ROUTING_KEY = os.getenv('PRODUCT_ROUTING_KEY')
SCHEDULE_ROUTING_KEY = os.getenv('SCHEDULE_ROUTING_KEY')

# Assuming you have a list of image paths and corresponding product IDs
product_ids = []  # list of product IDs
product_features = np.array([])


def setup_rabbitmq_consumer():
    try:
        connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
        channel = connection.channel()

        # Declare queues for product updates and scheduled tasks
        channel.queue_declare(queue=PRODUCT_UPDATES_QUEUE)
        channel.queue_declare(queue=SCHEDULE_QUEUE)

        # Declare exchanges
        channel.exchange_declare(exchange=PRODUCT_EXCHANGE, exchange_type='topic', durable=True)
        channel.exchange_declare(exchange=SCHEDULE_EXCHANGE, exchange_type='direct')

        # Product updates consumer
        channel.queue_bind(exchange=PRODUCT_EXCHANGE, queue=PRODUCT_UPDATES_QUEUE, routing_key=PRODUCT_ROUTING_KEY)
        channel.basic_consume(queue=PRODUCT_UPDATES_QUEUE, on_message_callback=handle_product_update, auto_ack=True)

        # Schedule consumer
        channel.queue_bind(exchange=SCHEDULE_EXCHANGE, queue=SCHEDULE_QUEUE, routing_key=SCHEDULE_ROUTING_KEY)
        channel.basic_consume(queue=SCHEDULE_QUEUE, on_message_callback=handle_schedule_trigger, auto_ack=True)

        print('Listening for messages...')
        channel.start_consuming()
    except Exception as e:
        print(f"Failed to connect to RabbitMQ server at {RABBITMQ_HOST}. Please check if the server is running and "
              f"the host is correct.")
        print(f"Error: {e}")

def handle_product_update(ch, method, properties, body):
    message = json.loads(body)
    product_id = message['id']
    images = message['images']
    action = message['op']
    if action == 'd':
        mongodb_handler.mark_for_deletion(product_id)
    else:
        mongodb_handler.save_for_later_processing(product_id, images, action)
    print(f"Processed update for product ID: {product_id} with action: {action}")


def handle_schedule_trigger(ch, method, properties, body):
    global product_ids, product_features

    # get product not trained
    products = mongodb_handler.get_untrained_products()
    count = products.count()
    features = []
    labels = []
    uuid = str(uuid.uuid4())
    for product in products:
        images = product['images']
        for img in images:
            img_path = f'{uuid}.jpg'  # Save the image temporarily
            img.save(img_path)
            features.append(load_and_preprocess_images(img_path))
            labels.append(product['product_id'])  # Assuming the label for each image is stored in the product
            os.remove(img_path)  # Clean up the temporary image

    features = np.vstack(features)
    labels = np.array(labels)
    # Train the model with the extracted features
    train_resnet50_model(features, labels, count)

    # mongodb_handler.update_product_as_trained([str(product['product_id']) for product in products])
    mongodb_handler.update_product_as_trained(products)
    print('Model trained successfully!')

    setup()


def get_product_id():
    global product_ids
    if len(product_ids) == 0:
        setup()
    return product_ids


def get_product_features():
    global product_features
    if len(product_features) == 0:
        setup()
    return product_features


def setup():
    global product_ids, product_features
    products = mongodb_handler.get_trained_products()
    product_ids = [str(product['product_id']) for product in products]
    product_features = np.array([product['images'] for product in products])


def load_and_preprocess_images(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
