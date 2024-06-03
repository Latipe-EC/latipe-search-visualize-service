# model_utils.py
import os

import numpy as np
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from keras.src.applications.resnet import ResNet50
from keras.src.callbacks import ModelCheckpoint
from keras.src.layers import GlobalAveragePooling2D, Dense
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.optimizers import Adam
from tensorflow.keras import Model

# model = load_model(weights_path='efficientnetb7_model.h5')

data_dir = 'img'
datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=20,  # NEW: random rotations
    width_shift_range=0.2,  # NEW: random width shifts
    height_shift_range=0.2,  # NEW: random height shifts
    horizontal_flip=True,
    validation_split=0.2)

# Define the generators for the training and validation sets
train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical',
    subset='training')  # set as training data

# Calculate class weights
from sklearn.utils import class_weight

class_weights = class_weight.compute_sample_weight(class_weight='balanced', y=train_generator.classes)

class_weights_dict = dict(enumerate(class_weights))


# Extract features using the trained model
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(600, 600))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array)
    features = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)).predict(img_preprocessed)
    return features.flatten()


EPOCHS = 20
BATCH_SIZE = 8

def train_resnet50_model(total, num_classes, epochs=EPOCHS, batch_size=BATCH_SIZE):

    base_model = ResNet50(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)  # Add a Dense layer for classification

    # Define a callback to save the model's weights
    checkpoint = ModelCheckpoint('resnet50_model.weights.h5', save_weights_only=True, save_best_only=True, monitor='val_loss',
                                 mode='min')

    # Compile the model
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Check if class_weights is defined and has the correct length
    if class_weights is not None and len(class_weights) == num_classes:
        class_weights_dict = dict(enumerate(class_weights))
    else:
        class_weights_dict = None

    # Fit the model
    history = model.fit(train_generator,
                        steps_per_epoch=total // batch_size, batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[checkpoint], verbose=1,
                        class_weight=class_weights_dict)  # Use class weights if available

    # Save the entire model to a Keras file after training
    model.save('resnet50_model.weights.keras')

    return history
