# app.py

from flask import Flask, request, jsonify
import os
import feature_extractor
import mongodb_handler

app = Flask(__name__)


@app.route('/search', methods=['POST'])
def search():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    image_path = 'temp_search_image.jpg'
    image_file.save(image_path)

    query_features = feature_extractor.extract_features(image_path)
    os.remove(image_path)  # Clean up the image file after processing

    top_product_ids = mongodb_handler.search_similar_features(query_features)
    return jsonify({'top_product_ids': top_product_ids})

