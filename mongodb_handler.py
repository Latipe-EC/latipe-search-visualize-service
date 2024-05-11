# rabbitmq_consumer.py

import os

import numpy as np
import requests
from pymongo import MongoClient

# Initialize MongoDB connection
mongo_uri = os.getenv('MONGO_URI', 'mongodb://localhost:27017/')
client = MongoClient(mongo_uri)
db = client[os.getenv('MONGO_DB_NAME', 'ecommerce')]
pretrain_product = db[os.getenv('PRETRAIN_COLLECTION', 'pretrain')]
trained_product = db[os.getenv('TRAINED_COLLECTION', 'trained')]
deleted_product = db[os.getenv('DELETED_COLLECTION', 'deleted')]
PRODUCT_SERVICE_URL = os.getenv('PRODUCT_SERVICE_URL')


def save_for_later_processing(product_id):
    response = requests.get(f'{PRODUCT_SERVICE_URL}/products-es/{product_id}', headers={
        "Content-Type": "application/json"
    })

    pretrain_product.update_one(
        {"product_id": product_id},
        {"$set": {"product_id": product_id, "images": response.json()['images']}},
        upsert=True
    )


def mark_for_deletion(product_id):
    if pretrain_product.find_one({"product_id": product_id}):
        pretrain_product.delete_one({"product_id": product_id})

    product = trained_product.find_one({"product_id": product_id})
    if product is not None:
        trained_product.delete_one({"product_id": product_id})
        deleted_product.insert_one(product)


def search_for_similar_images(query_features):
    results = trained_product.find({
        "trained": True
    })
    min_distance = float('inf')
    closest_product_id = None
    for product in results:
        stored_features = np.array(product['images'])
        distance = np.linalg.norm(query_features - stored_features)
        if distance < min_distance:
            min_distance = distance
            closest_product_id = product['_id']
    return closest_product_id


def get_untrained_products():
    return pretrain_product.find()


def get_trained_products():
    return trained_product.find()


# def update_product_as_trained(product_ids, f):
#     features_collection.update_many({
#         "product_id": {"$in": product_ids}
#     }, {"$set": {"trained": True}})


def update_product_as_trained(products):
    pretrain_product.delete_many({"product_id": {"$in": [product['product_id'] for product in products]}})
    for product in products:
        trained_product.update_one(
            {"product_id": product['product_id']},
            {"$set": {"product_id": product['product_id'], "images": product['images']}},
            upsert=True
        )
