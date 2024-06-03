# search_api.py

import os

from flask import Flask, request, jsonify
from pymongo import MongoClient
from sklearn.neighbors import NearestNeighbors

from model_utils import extract_features
from rabbitmq_consumer import get_product_id, get_product_features

app = Flask(__name__)

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['ecommerce']
features_collection = db['features']


@app.route('/search', methods=['POST'])
def search():
    # Check if an image file was uploaded
    if 'image' not in request.files:
        return jsonify({'error': 'No image file in request'}), 400

    img_file = request.files['image']

    # Define the allowed extensions
    allows = ['png', 'jpg', 'jpeg']

    # Check if the uploaded file's extension is in the allowed extensions
    if not any(img_file.filename.endswith('.' + ext) for ext in allows):
        return jsonify({'error': 'Invalid file type'}), 400

    # Save the uploaded image to a temporary file
    uuid = request.form['uuid']
    img_path = f'${uuid}.jpg'
    img_file.save(img_path)

    # Extract features from the uploaded image
    features = get_product_features()
    product_ids = get_product_id()

    # Get the product features and IDs from the database
    search_features = extract_features(img_path)
    os.remove(img_path)  # Clean up the temporary image

    # Fit the NearestNeighbors model with the product features
    nbr = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(features)

    # Find the 5 nearest neighbors
    distances, indices = nbr.kneighbors([search_features])

    # Get the product IDs of the nearest neighbors
    nearest_product_ids = [product_ids[i] for i in indices[0]]

    # Return the product IDs of the nearest neighbors
    return jsonify({'data': nearest_product_ids})
