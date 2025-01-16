import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

def extract_data():
    """Load and preprocess CIFAR-10 data."""
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test

def create_cnn(input_shape=(32, 32, 3), num_classes=10):
    """Define and return a CNN model."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def train_model(model, x_train, y_train, epochs=10, batch_size=64):
    """Compile and train the CNN model."""
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    return history

def save_model(model, save_path='data/model/cnn_cifar10.h5'):
    """Save the trained model to the specified path."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved at: {save_path}")

def plot_training(history, save_dir='data/plots'):
    """Plot and save training accuracy and loss."""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Save the plot
    plot_path = os.path.join(save_dir, 'training_plots.png')
    plt.savefig(plot_path)
    plt.show()
    print(f"Plots saved at: {plot_path}")

def evaluate_model(model, x_test, y_test):
    """Evaluate the model on the test dataset."""
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")