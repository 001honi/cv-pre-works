import cv2, timeit
import numpy as np
import matplotlib.pyplot as plt

from lib.custom_data_preprocessing import Dataset


# Loading Dataset & Preprocess via custom Dataset() class
#=======================================================================================
# Dataset Images are stored in the Dataset object as ds.imgs; 
# Labels as ds.labels; One-hot-Encoding as ds.labels_one_hot
print("\n\n(!) DATA Preprocessing")
TR_start= timeit.default_timer()
path = "./dataset/custom_dataset/"
color = 0 # 0: Gray 1: BGR 
print("Dataset is loading..")
ds = Dataset(path, color)
print("Rescaling images to 1..")
ds.normalize()
print("Shuffle process..")
ds.shuffle()
# print("One-hot-encoding..")
# ds.one_hot_encoding()
print("Shape of dataset:")
print(ds.imgs.shape)
print("Shape of one-input-img:")
print(ds.imgs[0].shape)
# End of Data Preprocessing
TR_stop = timeit.default_timer() 
print(f"(~) SUCCESSFULL | Execution Time: {TR_stop-TR_start:.3f} sec\n\n")



# Visualize Training Images
#=======================================================================================
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 8)
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

print(f"First 8 labels of dataset: {ds.labels[:8]}\n\n")
plotImages(ds.imgs[:8])



# Reshaping the Image Set for the Input Layer of the Network
#=======================================================================================
# Image shape is not convenient for the model input layer dimensions
# We need 1-more dimension representing channel length which is 1 
# print(ds.imgs.shape)      # (-1, H, W)
if not color:
    ds.add_channel_dim()    # (-1, H, W, 1)
    print("Channel Dimension is added for GRAY-SCALE img")
    print(ds.imgs[0].shape, end="\n\n")
IMG_H, IMG_W, CHANNEL = ds.imgs[0].shape



# Creating a Sequential CNN Model
#=======================================================================================
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# model = tf.keras.models.load_model(f'./models/hand-recog-mobile-net-v2-tf-1')

model = Sequential([
    Conv2D(16, 5, padding='same', activation='relu', input_shape=(IMG_H, IMG_W, CHANNEL)),
    MaxPooling2D(),
    Conv2D(32, 5, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 2, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 2, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dropout(0.3),
    Dense(1024, activation="relu"),
    Dropout(0.2),
    Dense(512, activation="relu"),
    Dropout(0.2),
    Dense(512, activation="relu"),
    Dropout(0.2),
    Dense(512, activation="relu"),
    Dropout(0.3),
    Dense(10)
])

model.summary()



# Transfer Learning via Mobile Net v2 Model
#=======================================================================================
# import tensorflow_hub as hub

# feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2" 

# feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
#                                          input_shape=(224,224,3))

# feature_extractor_layer.trainable = False

# model = Sequential([
#     feature_extractor_layer,
#     Dropout(0.2),
#     Dense(10)
# ])

# model.summary()



# Splitting Dataset for Training
#=======================================================================================
print("Dataset Splitting..")
ds.split_data(0.2)
print("Size of dataset:")
print(len(ds.imgs))
print("Size of training set:")
print(f"{len(ds.x_train)}")
print("Size of validation set:")
print(f"{len(ds.x_valid)}", end="\n\n")

assert len(ds.imgs) == len(ds.y_train)+len(ds.y_valid)



# Compile & Train the Model
#=======================================================================================
# model = tf.keras.models.load_model(f'./models/hand-recog-cnn-model-1')  # Loading Trained Model
# loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)    # Requies one-hot-encoding
TR_start= timeit.default_timer()

epochs = 15
batch_size = 64

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Fitting with basic model.fit() command
#----------------------------------------------
# history = model.fit(ds.imgs, ds.labels, validation_split=0.2, epochs=epochs)            
# history = model.fit(ds.x_train, ds.y_train, epochs=epochs)            


# Fitting with Image Augmentation
#----------------------------------------------
imgAug = ImageDataGenerator(width_shift_range=0.2,
                             height_shift_range=0.2,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)

history = model.fit(imgAug.flow(ds.x_train, ds.y_train, batch_size=batch_size),
                                 steps_per_epoch=len(ds.x_train)//batch_size,
                                 epochs=epochs,
                                 validation_data=(ds.x_valid, ds.y_valid),
                                 shuffle=1)

# End of Training
TR_stop = timeit.default_timer()
print(f"(~) SUCCESSFULL | Training Time: {TR_stop-TR_start:.3f} sec")



# Saving
#=======================================================================================
# model_ID = 'hand-recog-mobile-net-v2-tf-1'
model_ID = 'hand-recog-cnn-model-3'
model.save(f'./models/{model_ID}')



# Visualize training results
#=======================================================================================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()