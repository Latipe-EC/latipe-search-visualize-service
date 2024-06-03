import os
import random
import shutil
import time
from io import BytesIO

import pika
import requests
from PIL import Image
from dotenv import load_dotenv

import mongodb_handler
from rabbitmq_consumer import get_detail_product

# Load the .env file
load_dotenv()

# SCHEDULE_EXCHANGE = os.getenv('SCHEDULE_EXCHANGE')
# SCHEDULE_ROUTING_KEY = os.getenv('SCHEDULE_ROUTING_KEY')
# RABBITMQ_HOST = os.getenv('RABBITMQ_HOST')
#
# connection = pika.BlockingConnection(pika.ConnectionParameters(RABBITMQ_HOST))
# channel = connection.channel()
#
# channel.basic_publish(exchange=SCHEDULE_EXCHANGE,
#                       routing_key=SCHEDULE_ROUTING_KEY,
#                       body='train')

products = mongodb_handler.get_untrained_products()
products = get_detail_product([str(product['product_id']) for product in products])
imgs = {}

file_img = []
file_name = []

with open('download_records.txt', 'r') as record_file:
    for line in record_file:
        file_img.append(line.strip())
        file_name.append(line.strip().split('^^^')[1])

file_name.pop()


def download_image(url, folder, filename):
    if not os.path.exists(f'{os.getcwd()}/img/{folder}'):
        os.makedirs(f'{os.getcwd()}/img/{folder}')
    response = requests.get(url)

    if response.status_code != 200:
        print(f"Failed to fetch image: {url}")
        return

    try:
        # Open the image file
        img_down = Image.open(BytesIO(response.content))

        # Convert the image to RGB format
        img_down = img_down.convert('RGB')

        # Resize the image to 224x224 pixels
        img_down = img_down.resize((224, 224))

        # Save the processed image back to the file
        img_down.save(f'{os.getcwd()}/img/{folder}/{filename}', 'JPEG')
    except OSError:
        print(f"Image file {url} is truncated or corrupted. Skipping this file.")


############################################################
# Preprocess the images in the 'train' folder

# Define the path to the 'train' folder
# train_folder_path = 'img'
#
#
# def get_folder_names(directory):
#     return [name for name in os.listdir(directory) if
#             os.path.isdir(os.path.join(directory, name))]
#
#
# def get_file_names(directory):
#     return [name for name in os.listdir(directory) if
#             os.path.isfile(os.path.join(directory, name))]
#
#
# for folder_name in get_folder_names('img'):
#     for file_name in get_file_names(f'img/{folder_name}'):
#         img = Image.open(f'img/{folder_name}/{file_name}')
#         # Convert the image to RGB format
#         img = img.convert('RGB')
#
#         # Resize the image to 224x224 pixels
#         img = img.resize((224, 224))
#
#         # Save the processed image back to the file
#         print(f"Saving processed image: {file_name}")
#         img.save(f'img/{folder_name}/{file_name}')

################################################################################


for product in products:
    if product['id'] in file_name:
        print('Skip product: ', product['id'])
        continue

    images = product['images']
    print('Handle for product: ', product['id'])
    sleep_time = random.randint(5, 10)  # Generate a random integer between 1 and 5
    print(f"Sleeping for {sleep_time} seconds")
    time.sleep(sleep_time)  # Sleep for the generated duration
    for img in images:
        if f"{img}^^^{product['id']}" in file_img:
            continue
        if img not in imgs:
            imgs[img] = product['id']
            img_name = img.split('/')[-1]
            download_image(img, product['id'], img_name)
            # Save the record to a file
            with open('download_records.txt', 'a') as record_file:  # 'a' mode for append
                record_file.write(f"{img}^^^{product['id']}\n")
        else:
            if not os.path.exists(f"{os.getcwd()}/img/{product['id']}"):
                os.makedirs(f"{os.getcwd()}/img/{product['id']}")

            shutil.copy(f"{os.getcwd()}/img/{imgs.get(img)}/{img.split('/')[-1]}"
                        , f"{os.getcwd()}/img/{product['id']}/{img.split('/')[-1]}")
            # Save the record to a file
            with open('download_records.txt', 'a') as record_file:  # 'a' mode for append
                record_file.write(f"{img}^^^{product['id']}\n")
