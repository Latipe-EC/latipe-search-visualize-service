# model_utils.py

import numpy as np
from keras import Model
from keras.src.applications.efficientnet import EfficientNetB7, preprocess_input
from keras.src.callbacks import ModelCheckpoint, CSVLogger
from keras.src.layers import GlobalAveragePooling2D, Dense
from keras.src.optimizers import Adam
from keras.src.saving import load_model
from keras.preprocessing import image
from keras_applications.resnet50 import ResNet50

# model = load_model(weights_path='efficientnetb7_model.h5')


# Extract features using the trained model
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(600, 600))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_array)
    features = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3)).predict(img_preprocessed)
    return features.flatten()


# def get_callbacks():
#     # Saves the model after each epoch
#     checkpoint = ModelCheckpoint(
#         'efficientnetb7_model.h5', save_best_only=True, monitor='val_loss', mode='min'
#     )
#     # Logs epoch, accuracy, loss, val_accuracy, val_loss
#     csv_logger = CSVLogger('training_log.csv', append=True)
#
#     return [checkpoint, csv_logger]
#
#
# def train_model(images, labels, epochs=10, batch_size=32):
#     callbacks = get_callbacks()
#     history = model.fit(images, labels, batch_size=batch_size, epochs=epochs, callbacks=callbacks, validation_split=0.1)
#     return history


def train_resnet50_model(images, labels, num_classes, epochs=10, batch_size=32):
    # Load ResNet50 as the base model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)  # Add a Dense layer for classification

    # Define a callback to save the model's weights
    checkpoint = ModelCheckpoint('resnet50_weights.h5', save_weights_only=True, save_best_only=True, monitor='val_loss',
                                 mode='min')

    # Compile the model
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Fit the model
    history = model.fit(images, labels, batch_size=batch_size, epochs=epochs, callbacks=[checkpoint],
                        validation_split=0.1)

    return history