# Import Libraries
import tensorflow as tf 
from sklearn.model_selection import train_test_split

# Load Data
mnist = tf.keras.datasets.mnist
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2)

# Preprocess Data 
X_train, X_test, X_val = X_train / 255.0, X_test / 255.0, X_val / 255.0

# Build Dataset 
batch_size = 100

X_train_new_axis = X_train[..., tf.newaxis]
X_val_new_axis = X_val[..., tf.newaxis]
X_test_new_axis = X_test[..., tf.newaxis]

shuffle_size = 100000

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_new_axis, Y_train)).shuffle(shuffle_size).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val_new_axis, Y_val)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_new_axis, Y_test)).batch(batch_size)

# Model Engineering
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics

import numpy as np

# Set Hyperparameters
hidden_size = 256
output_dim = 10
EPOCHS = 30
learning_rate = 1e-3 

class CNN_Model(Model):
    def __init__(self, hidden_size, output_dim):
        super(CNN_Model, self).__init__()

        self.conv1 = Conv2D(filters = 64, kernel_size = 3, activation='relu', padding = 'SAME')
        self.maxpool2d1 = MaxPool2D(padding = 'SAME')
        self.conv2 = Conv2D(filters = 128, kernel_size = 3, activation='relu', padding = 'SAME')
        self.maxpool2d2 = MaxPool2D(padding = 'SAME')
        self.conv3 = Conv2D(filters = 256, kernel_size = 3, activation='relu', padding = 'SAME')
        self.maxpool2d3 = MaxPool2D(padding = 'SAME')
        self.flatten = Flatten()
        self.d1 = Dense(hidden_size, activation = 'relu')
        self.d2 = Dropout(0.2)
        self.d3 = Dense(output_dim, activation = 'softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool2d1(x)
        x = self.conv2(x)
        x = self.maxpool2d2(x)
        x = self.conv3(x)
        x = self.maxpool2d3(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        y = self.d3(x)
        return y 

model = CNN_Model(hidden_size, output_dim)

# Optimizer 
optimizer = optimizers.Adam(learning_rate = learning_rate)

# Define Loss Function
criteria = losses.SparseCategoricalCrossentropy()
train_loss = metrics.Mean(name = 'train_loss')
train_accuracy = metrics.SparseCategoricalAccuracy(name = 'train_accuracy')

val_loss = metrics.Mean(name = 'val_loss')
val_accuracy = metrics.SparseCategoricalAccuracy(name = 'val_accuracy')

test_loss = metrics.Mean(name = 'test_loss')
test_accuracy = metrics.SparseCategoricalAccuracy(name = 'test_accuracy')

# Training
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = criteria(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

def val_step(images, labels):
    predictions = model(images)
    loss = criteria(labels, predictions)

    val_loss(loss)
    val_accuracy(labels, predictions)

def test_step(images, labels):
    predictions = model(images)
    loss = criteria(labels, predictions)

    test_loss(loss)
    test_accuracy(labels, predictions)

from tqdm import tqdm, notebook, trange

for epoch in range(EPOCHS):
    with notebook.tqdm(total = len(train_dataset), desc = f'Epoch {epoch + 1}', position = 0) as pbar:
        train_losses = []
        train_accuracies = []
        for images, labels in train_dataset:
            train_step(images, labels)
            loss_val = train_loss.result()
            acc = train_accuracy.result()

            train_losses.append(loss_val)
            train_accuracies.append(acc)

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.4f} ({np.mean(train_losses):.4f}) Acc: {acc:.3f} ({np.mean(train_accuracies):.3f})")
