# Import the require modules
import numpy as np
import tensorflow as tf
import os
import cv2

# This will be used to display progress bar for the iterations
from tqdm import tqdm

# List all the folders inside the dataser folder using os module
files = os.listdir("../Dataset for Face Detection")
# Check if the folders are successfully accessed
# print(files)

# Store the images and their corresponding labels inside lists for the time-being
images=[]
labels=[]
path="../Dataset for Face Detection/"

# Loop through each dataset
for i in range(len(files)):
    # Access each dataset folder
    folder = os.listdir(path + files[i])
    # Now access the cropped images folder
    sub_folder = os.listdir(path + files[i] + "/" + folder[0])
    # Loop through each image inside the cropped folder
    for k in tqdm(range(len(sub_folder))):
        try:
            # Start reading the images of extracted faces inside the folder
            img = cv2.imread(path + files[i] + "/" + folder[0] + "/" + sub_folder[k])
            print(path + files[i] + "/" + folder[0] + "/" + sub_folder[k])
            # Convert the image to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Resize the image to (96,96)
            img = cv2.resize(img,(96,96))
            # Store the images in the list
            images.append(img)
            labels.append(i)
        except:
            pass

# Check if the images and labels are loaded succesfully inside our lists
# print(images)
# print(labels)

# Optional: Clear your ram memory
# import gc
# gc.collect()

# Now, we will convert the image list to a numpy array
# Also divide image by 225 to scale image from 0-1 range
images_array = np.array(images)/255.0
labels_array = np.array(labels)

# Let's split and shuffle our dataset for training and testing (85-15 split)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(images_array, labels_array, test_size=0.15)

# Finally, let's create a model, We have decided to employ image classification model
from keras import layers, callbacks, utils, applications, optimizers
from keras.models import Sequential, Model, load_model

# Create the model
model = Sequential()
# Employing the EfficientNet pre-trained model
pre_trained_model = tf.keras.applications.EfficientNetB0(input_shape=(96,96,3), include_top = False, weights="imagenet")

# Add this model to our model
model.add(pre_trained_model)
# Add some layer to improve accuracy
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dropout(0.3))

# Add dense layer with number of Output
model.add(layers.Dense(1))
# Output is 1 and the values are fro 0-5
# Let's see the summary of our model
model.summary()

# Compile model
model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])

# Create a checkpoint to save best accuracy model
# Define the directory where you want to save the checkpoint
checkpoint_dir = "trained_model"
os.makedirs(checkpoint_dir, exist_ok=True)

# Define the checkpoint filepath with the correct extension
ckp_path = os.path.join(checkpoint_dir, "model.weights.h5")
model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=ckp_path, monitor="val_mae", mode="auto", save_best_only=True, save_weights_only=True)
# Explanation of above line of code:
# 1. watch mae of test set and when it decreases, save the model
# 2. mode is used to check increase or decrease in mae
# 3. if it's true then save the weights only

# Create a lr reducer which will decrease learning rate when accuracy doesn't increase
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.9, monitor="val_mae", mode="auto", cooldown=0, patience=5, verbose=1, min_lr=1e-6)
# Explanation of above line of code:
# 1. patience: wait for 5 epoch then decrease the learning rate
# 2. verbose: show accuracy(val_mae) every epoch
# 3. minimum learning ratee is set at 10^-6

# Now, let's finally train the model
Epoch = 300
Batch_Size = 64
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size = Batch_Size, epochs = Epoch, callbacks = [model_checkpoint, reduce_lr])

# Once, the training is finished, load the best model
model.load_weights(ckp_path)

# Convert best model to tensorflow lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model=converter.convert()
# Save the model
with open("model.tflite","wb") as f:
    f.write(tflite_model)

# Let's observe the predictions on test set
prediction_val = model.predict(X_test, batch_size=64)
prediction_val[:3]

# Original label
y_test[:3]
