# rabbitmq_consumer.py
import json
import os
import uuid
from io import BytesIO

import numpy as np
import pika
import requests
from PIL import Image
from dotenv import load_dotenv
from keras.preprocessing import image

import mongodb_handler
from model_utils import train_resnet50_model

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
PRODUCT_SERVICE_URL = os.getenv('PRODUCT_SERVICE_URL')

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
        channel.basic_publish(exchange=SCHEDULE_EXCHANGE,
                              routing_key=SCHEDULE_ROUTING_KEY,
                              body='train')
        print('Listening for messages...')
        channel.start_consuming()
    except Exception as e:
        print(f"Failed to connect to RabbitMQ server at {RABBITMQ_HOST}. Please check if the server is running and "
              f"the host is correct.")
        print(f"Error: {e}")


def handle_product_update(ch, method, properties, body):
    message = json.loads(body)
    product_id = message['id']
    action = message['op']
    print(f"Received message: [product_id: {product_id}]")

    if action == 'd':
        mongodb_handler.mark_for_deletion(product_id)
    else:
        mongodb_handler.save_for_later_processing(product_id)
    print(f"Processed update for product ID: {product_id} with action: {action}")


def get_uuid():
    return str(uuid.uuid4())


def handle_schedule_trigger(ch, method, properties, body):
    global product_ids, product_features

    # get product not trained
    # products = mongodb_handler.get_untrained_products()
    # products = get_detail_product([str(product['product_id']) for product in products])
    # count_trained = mongodb_handler.count_trained_products()
    # count = len(products) + count_trained
    #
    # features = []
    # labels = []
    # for product in products:
    #     images = product['images']
    #     print('Handle for product: ', product['id'])
    #     for img in images:
    #         features.append(load_and_preprocess_images(img))
    #         labels.append(product['id'])  # Assuming the label for each image is stored in the product
    #
    # features = np.vstack(features)
    # labels = np.array(labels)
    # # Train the model with the extracted features
    # print('Starting train model...!')
    # train_resnet50_model(features, labels, count)
    #
    # # mongodb_handler.update_product_as_trained([str(product['product_id']) for product in products])
    # mongodb_handler.update_product_as_trained(products)
    # print('Model trained successfully!')
    #
    # setup()

    features = []
    labels = []

    # Train the model with the extracted features
    print('Starting train model...!')
    count = 0
    # for folder_name in get_folder_names('img'):
    #     for img_name in get_file_names('img/' + folder_name):
    #         img_path = 'img/' + folder_name + '/' + img_name
    #         img = Image.open(img_path)
    #         img = img.convert('RGB')  # Convert image to RGB format
    #         img = img.resize((224, 224))  # Resize the image
    #         img_array = image.img_to_array(img)
    #         img_array = np.expand_dims(img_array, axis=0)
    #         features.append(img_array)
    #         labels.append(folder_name)
    #         count += 1
    #
    # features = np.vstack(features)
    # labels = np.array(labels)

    total = 0
    for folder_name in get_folder_names('img'):
        total += len(get_file_names('img/' + folder_name))

    # Train the model with the extracted features
    print('Load dataset successfully!')
    train_resnet50_model(total,
                         len(get_folder_names('img')))

    # mongodb_handler.update_product_as_trained([str(product['product_id']) for product in products])
    # mongodb_handler.update_product_as_trained(products)
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


def load_and_preprocess_images(url):
    img = load_image_from_url(url)
    img = img.convert('RGB')  # Convert image to RGB format
    img = img.resize((224, 224))  # Resize the image
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def get_detail_product(product_ids):
    response = requests.post(f'{PRODUCT_SERVICE_URL}/products-es-multiple', headers={
        "Content-Type": "application/json"
    }, json={"product_ids": product_ids})

    if response.status_code != 200:
        print("Failed to fetch product details")
        return None

    return response.json()


def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))


def download_image(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)


def get_folder_names(directory):
    return [name for name in os.listdir(directory) if
            os.path.isdir(os.path.join( directory, name))]


def get_file_names(directory):
    return [name for name in os.listdir(directory) if
            os.path.isfile(os.path.join(directory, name))]
