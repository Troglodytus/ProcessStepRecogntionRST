import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from IPython.display import clear_output
from tensorflow.keras.models import load_model
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    if not title:
        title = 'Confusion matrix'
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
    plt.show()
    fig.savefig('model_confusionmatrix.png')

# class ShowAugmentedImageCallback(Callback):
#     def __init__(self, dataset, num_images=1):
#         self.dataset = dataset
#         self.num_images = num_images

#     def on_epoch_end(self, epoch, logs=None):
#         # Display a random image from the dataset at the end of each epoch
#         for images, _ in self.dataset.shuffle(1024).batch(self.num_images).take(1):
#             for i in range(self.num_images):
#                 plt.figure(figsize=(3, 3))
#                 plt.imshow(images[i].numpy())
#                 plt.axis('off')
#                 plt.show()

class LiveLossPlot(Callback):
    def on_train_begin(self, logs=None):
        plt.ion()  # Turn on interactive mode
        
        # Initialize lists to store the epochs, loss, and accuracy
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.train_accuracy = []
        self.val_accuracy = []

        # Set up the subplots
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 5))
        

    def on_epoch_end(self, epoch, logs=None):
        # Append the logs to the lists
        self.epochs.append(epoch+1)
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.train_accuracy.append(logs.get('sparse_categorical_accuracy'))
        self.val_accuracy.append(logs.get('val_sparse_categorical_accuracy'))

        # Loss plot
        self.ax1.clear()
        self.ax1.plot(self.epochs, self.train_loss, label='Training Loss')
        self.ax1.plot(self.epochs, self.val_loss, label='Validation Loss')
        self.ax1.set_title('Training and Validation Loss')
        self.ax1.set_xlabel('Epoch')
        self.ax1.set_ylabel('Loss')
        self.ax1.legend()
        self.ax1.grid(True)

        # Accuracy plot
        self.ax2.clear()
        self.ax2.plot(self.epochs, self.train_accuracy, label='Training Accuracy')
        self.ax2.plot(self.epochs, self.val_accuracy, label='Validation Accuracy')
        self.ax2.set_title('Training and Validation Accuracy')
        self.ax2.set_xlabel('Epoch')
        self.ax2.set_ylabel('Accuracy')
        self.ax2.legend()
        self.ax2.grid(True)

        # Draw the figure and pause to update
        self.fig.canvas.draw()
        plt.pause(0.5)

    def on_train_end(self, logs=None):
        plt.ioff()  # Turn off interactive mode
        plt.show()
        # Save the figure
        self.fig.savefig('model_accuracyloss.png')


class PrintLearningRate(Callback):
    def on_epoch_end(self, epoch, logs=None):   
        lr = self.model.optimizer._decayed_lr(tf.float32)  # Corrected line
        print(f'\nLearning rate for epoch {epoch + 1} is {lr.numpy()}')


# Load the labels CSV file
labels_df = pd.read_csv('labels7class.csv')
#load model
#pre_trained_model = load_model('RST5C_7Class_Model_v05.h5')
pre_trained_model = load_model('RST_7Class_256_Aug_Model_v03.h5')
pre_trained_model.summary()

# Define the label mapping
label_map = {}
for i, label in enumerate(labels_df['Label'].unique()):
    label_map[i] = label

# Create a list of filepaths by combining the 'Filepath' and 'Filename' columns
filepaths = [os.path.join(file_path, filename) for file_path, filename in zip(labels_df['Filepath'], labels_df['Filename'])]
pfad = "C:/Users/Eichleitner/Documents/Coding/7class"

# Define input dimensions
input_height = 256
input_width = 256
num_channels = 3
num_output_nodes = len(label_map)

# Define CNN architecture
num_input_nodes = 32
num_hidden_nodes = 32
kernel_size = 3

# x = keras.Input(shape=(input_height, input_width, num_channels))
# conv1 = keras.layers.Conv2D(filters=num_input_nodes, kernel_size=kernel_size, activation=tf.nn.relu, data_format='channels_last', weights=pre_trained_model.layers[1].get_weights())(x)
# pool1 = keras.layers.MaxPooling2D(pool_size=2, strides=2, data_format='channels_last')(conv1)
# conv2 = keras.layers.Conv2D(filters=num_hidden_nodes*2, kernel_size=kernel_size, activation=tf.nn.relu, weights=pre_trained_model.layers[3].get_weights())(pool1)
# pool2 = keras.layers.MaxPooling2D(pool_size=2, strides=2)(conv2)
# conv3 = keras.layers.Conv2D(filters=num_hidden_nodes*4, kernel_size=kernel_size, activation=tf.nn.relu, weights=pre_trained_model.layers[5].get_weights())(pool2)
# pool3 = keras.layers.MaxPooling2D(pool_size=2, strides=2)(conv3)
# conv4 = keras.layers.Conv2D(filters=num_hidden_nodes*8, kernel_size=kernel_size, activation=tf.nn.relu, weights=pre_trained_model.layers[7].get_weights())(pool3)
# pool4 = keras.layers.MaxPooling2D(pool_size=2, strides=2)(conv4)
# conv5 = keras.layers.Conv2D(filters=num_hidden_nodes*8, kernel_size=kernel_size, activation=tf.nn.relu, kernel_initializer='he_normal')(pool4)
# pool5 = keras.layers.MaxPooling2D(pool_size=2, strides=2)(conv5)
# flatten = keras.layers.Flatten()(pool5)
# dense1 = keras.layers.Dense(units=num_hidden_nodes*8, activation=tf.nn.relu)(flatten)
# dropout = keras.layers.Dropout(rate=0.25)(dense1)
# dense2 = keras.layers.Dense(units=num_hidden_nodes*4, activation=tf.nn.relu)(dropout)
# dense3 = keras.layers.Dense(units=num_hidden_nodes*2, activation=tf.nn.relu)(dense2)
# output = keras.layers.Dense(units=num_output_nodes, activation='softmax')(dense3)

x = keras.Input(shape=(input_height, input_width, num_channels))
conv1 = keras.layers.Conv2D(filters=num_input_nodes, kernel_size=kernel_size, activation=tf.nn.relu, data_format='channels_last', weights=pre_trained_model.layers[1].get_weights())(x)
pool1 = keras.layers.MaxPooling2D(pool_size=2, strides=2, data_format='channels_last')(conv1)
conv2 = keras.layers.Conv2D(filters=num_hidden_nodes*2, kernel_size=kernel_size, activation=tf.nn.relu, weights=pre_trained_model.layers[3].get_weights())(pool1)
pool2 = keras.layers.MaxPooling2D(pool_size=2, strides=2)(conv2)
conv3 = keras.layers.Conv2D(filters=num_hidden_nodes*4, kernel_size=kernel_size, activation=tf.nn.relu, weights=pre_trained_model.layers[5].get_weights())(pool2)
pool3 = keras.layers.MaxPooling2D(pool_size=2, strides=2)(conv3)
conv4 = keras.layers.Conv2D(filters=num_hidden_nodes*8, kernel_size=kernel_size, activation=tf.nn.relu, weights=pre_trained_model.layers[7].get_weights())(pool3)
pool4 = keras.layers.MaxPooling2D(pool_size=2, strides=2)(conv4)
conv5 = keras.layers.Conv2D(filters=num_hidden_nodes*16, kernel_size=kernel_size, activation=tf.nn.relu, weights=pre_trained_model.layers[9].get_weights())(pool4)
pool5 = keras.layers.MaxPooling2D(pool_size=2, strides=2)(conv5)
flatten = keras.layers.Flatten()(pool5)
dense1 = keras.layers.Dense(units=num_hidden_nodes*16, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.001))(flatten)
dropout = keras.layers.Dropout(rate=0.25)(dense1)
dense2 = keras.layers.Dense(units=num_hidden_nodes*8, activation=tf.nn.relu)(dropout)
dense3 = keras.layers.Dense(units=num_hidden_nodes*4, activation=tf.nn.relu)(dense2)
output = keras.layers.Dense(units=num_output_nodes, activation='softmax')(dense3)

model = keras.Model(inputs=x, outputs=output)

# Define the activation model
activation_model = keras.Model(inputs=model.input, outputs=[layer.output for layer in model.layers if "conv2d" in layer.name])

# Define the learning rate scheduler
initial_learning_rate = 0.001
#initial_learning_rate = 0.008
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=10000,
    #decay_rate=0.05,
    decay_rate=0.033,
    staircase=True)
print_lr = PrintLearningRate()



# Compile the model with the optimizer and learning rate scheduler
model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
              loss=keras.losses.SparseCategoricalCrossentropy(),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])

# Define the training parameters
batch_size = 32
num_epochs_freezed = 1
num_epochs = 5
num_epochs_aug = 25

# Define the checkpoint directory and the checkpoint path
checkpoint_dir = "training_checkpoints"
checkpoint_path = os.path.join(checkpoint_dir, "cp-{epoch:04d}.ckpt")

# Ensure the checkpoint directory exists
os.makedirs(checkpoint_dir, exist_ok=True)

# Create a callback that saves the model's weights at the end of every epoch
cp_callback = ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=False,  # Set to False to save the full model
    save_freq='epoch'  # Save after each epoch
)

# Split the data into training and validation datasets
train_ds = tf.keras.utils.image_dataset_from_directory(
    pfad,
    validation_split=0.23,
    labels='inferred',
    label_mode='int',
    subset="training",
    seed=123,
    image_size=(input_height, input_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    pfad,
    validation_split=0.23,
    labels='inferred',
    label_mode='int',
    subset="validation",
    seed=123,
    image_size=(input_height, input_width),
    batch_size=batch_size)


################### FREEZING LAYERS  ###########################

# # Unfreeze all layers initially
# for layer in model.layers:
#     layer.trainable = True
# #freeze first convolutional layers (up to conv4)
# model.layers[1].trainable = False
# model.layers[2].trainable = False
# model.layers[3].trainable = False
# model.layers[4].trainable = False
# model.layers[5].trainable = False
# model.layers[6].trainable = False
# model.layers[7].trainable = False


# # Normalize the image pixels to 0-1
# train_ds = train_ds.map(lambda x, y: (x/255.0, y))
# val_ds = val_ds.map(lambda x, y: (x/255.0, y))

# # Train the model on the original dataset
# for epoch in range(num_epochs_freezed):
#     # Shuffle the indices of the images and labels
#     indices = np.random.permutation(len(labels_df))
#     df = labels_df.iloc[indices]

#     # Iterate over the batches of images and labels
#     for batch_images, batch_labels in train_ds:
#         # Train the model on the current batch of images and labels
#         model.train_on_batch(batch_images, batch_labels, reset_metrics=True)

#         # Print the current epoch, batch number, loss, and accuracy
#         batch_loss, batch_accuracy = model.evaluate(batch_images, batch_labels, verbose=0)
#         print(f"Epoch {epoch+1}/{num_epochs_freezed}, Loss: {batch_loss:.4f}, Accuracy: {batch_accuracy:.4f}")
#         pred_labels = model.predict(batch_images)
#         actual_labels = [label_map[label.numpy()] for label in batch_labels]
#         pred_labels = [label_map[np.argmax(label)] for label in pred_labels]
#         for i in range(len(batch_labels)):
#             print(f"Actual label: {actual_labels[i]}, Predicted label: {pred_labels[i]}")
#         # Evaluate the model on the validation dataset
#         for val_images, val_labels in val_ds:
#             val_loss, val_accuracy = model.evaluate(val_images, val_labels, verbose=0)
#             print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
#         lr = model.optimizer._decayed_lr(tf.float32).numpy()
#         print(f"Learning rate epoch {epoch+1} is {lr:.5f}")



################### Unfreeze all layers ######################
# for layer in model.layers:
#     layer.trainable = True

# Normalize the image pixels to 0-1
train_ds = train_ds.map(lambda x, y: (x/255.0, y))
val_ds = val_ds.map(lambda x, y: (x/255.0, y))

# Train the model on the original dataset
for epoch in range(num_epochs):
    # Shuffle the indices of the images and labels
    indices = np.random.permutation(len(labels_df))
    df = labels_df.iloc[indices]

    # Iterate over the batches of images and labels
    for batch_images, batch_labels in train_ds:
        # Train the model on the current batch of images and labels
        model.train_on_batch(batch_images, batch_labels, reset_metrics=True)

        # Print the current epoch, batch number, loss, and accuracy
        batch_loss, batch_accuracy = model.evaluate(batch_images, batch_labels, verbose=0)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {batch_loss:.4f}, Accuracy: {batch_accuracy:.4f}")
        pred_labels = model.predict(batch_images)
        actual_labels = [label_map[label.numpy()] for label in batch_labels]
        pred_labels = [label_map[np.argmax(label)] for label in pred_labels]
        for i in range(len(batch_labels)):
            print(f"Actual label: {actual_labels[i]}, Predicted label: {pred_labels[i]}")
        # Evaluate the model on the validation dataset
        for val_images, val_labels in val_ds:
            val_loss, val_accuracy = model.evaluate(val_images, val_labels, verbose=0)
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        lr = model.optimizer._decayed_lr(tf.float32).numpy()
        print(f"Learning rate epoch {epoch+1} is {lr:.5f}")


############### Data augmentation ############################
data_augmentation = keras.Sequential([
    #keras.layers.experimental.preprocessing.RandomFlip("horizontal"),
    keras.layers.experimental.preprocessing.RandomRotation(factor=0.05),
    keras.layers.experimental.preprocessing.RandomContrast(factor=0.04),
    keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.05, width_factor=0.05),
    #keras.layers.RandomBrightness(factor=0.02),
])
# Apply data augmentation to the training dataset
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))
val_ds = val_ds.map(lambda x, y: (data_augmentation(x, training=False), y))

live_loss_plot = LiveLossPlot()
#show_image_callback = ShowAugmentedImageCallback(train_ds)

# Train the model with the callback and data augmentation
history = model.fit(
    train_ds, 
    epochs=num_epochs_aug, 
    validation_data=val_ds, 
    callbacks=[cp_callback, print_lr, LiveLossPlot()]  # Add the checkpoint callback here
)

# Save the model in the input folder
model.save('model.h5')
# Plot the model architecture
tf.keras.utils.plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)


##########Validation and Metrics##############################
y_true = []
y_pred = []

# Iterate over the validation dataset
for images, labels in val_ds:
    # Make predictions
    preds = model.predict(images)
    preds = np.argmax(preds, axis=1)

    # Store true labels and predictions
    y_true.extend(labels.numpy())
    y_pred.extend(preds)

# Convert lists to numpy arrays
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Plot the confusion matrix
plot_confusion_matrix(y_true, y_pred, classes=list(label_map.values()))

# Generate the classification report
report = classification_report(y_true, y_pred, target_names=list(label_map.values()))
print(report)


# Write the classification report to a text file
with open('model_metrics.txt', 'w') as f:
    f.write(report)