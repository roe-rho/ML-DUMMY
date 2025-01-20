import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import logging  # New import

# Configure logging
logging.basicConfig(
    filename='training.log',  # Log file name
    level=logging.INFO,       # Logging level
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

def extract_data():
    """Load and preprocess CIFAR-10 data."""
    try:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)
        logger.info("Data loaded and preprocessed.")
        return x_train, y_train, x_test, y_test
    except Exception as e:
        logger.error("Error in data extraction: %s", str(e))
        raise

def create_cnn(input_shape=(32, 32, 3), num_classes=10):
    """Define and return an enhanced CNN model with improved parameters for better accuracy."""
    try:
        model = models.Sequential([
            # First convolutional block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Second convolutional block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),

            # Third convolutional block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),

            # Flatten and fully connected layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])

        logger.info("Enhanced CNN model created.")
        return model
    except Exception as e:
        logger.error("Error in CNN creation: %s", str(e))
        raise
    # def create_cnn(input_shape=(32, 32, 3), num_classes=10):
    #     """Define and return a CNN model."""
    #     try:
    #         model = models.Sequential([
    #             layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
    #             layers.MaxPooling2D((2, 2)),
    #             layers.Conv2D(64, (3, 3), activation='relu'),
    #             layers.MaxPooling2D((2, 2)),
    #             layers.Conv2D(64, (3, 3), activation='relu'),
    #             layers.Flatten(),
    #             layers.Dense(64, activation='relu'),
    #             layers.Dense(num_classes, activation='softmax')
    #         ])
    #         logger.info("CNN model created.")
    #         return model
    #     except Exception as e:
    #         logger.error("Error in CNN creation: %s", str(e))
    #         raise

def save_model_summary(model, save_path='data/model_summary.txt'):
    """Save the model summary to a text file."""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        logger.info(f"Model summary saved at: {save_path}")
    except Exception as e:
        logger.error("Error in saving model summary: %s", str(e))
        raise

def train_model(model, x_train, y_train, epochs=10, batch_size=64):
    """Compile and train the CNN model."""
    try:
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        logger.info("Model training completed.")
        return history
    except Exception as e:
        logger.error("Error in training the model: %s", str(e))
        raise

def save_model(model, save_path='data/model/cnn_cifar10.h5'):
    """Save the trained model to the specified path."""
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        logger.info(f"Model saved at: {save_path}")
    except Exception as e:
        logger.error("Error in saving model: %s", str(e))
        raise

def plot_training(history, save_dir='data/plots'):
    """Plot and save training accuracy and loss."""
    try:
        os.makedirs(save_dir, exist_ok=True)
        plt.figure(figsize=(12, 4))

        # Accuracy
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Save the plot
        plot_path = os.path.join(save_dir, 'training_plots.png')
        plt.savefig(plot_path)
        plt.show()
        logger.info(f"Training plots saved at: {plot_path}")
    except Exception as e:
        logger.error("Error in plotting training results: %s", str(e))
        raise

def evaluate_model(model, x_test, y_test):
    """Evaluate the model on the test dataset."""
    try:
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
        logger.info(f"Test Loss: {test_loss}")
        logger.info(f"Test Accuracy: {test_accuracy}")
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")  # Print accuracy to console
    except Exception as e:
        logger.error("Error in model evaluation: %s", str(e))
        raise

def plot_confusion_matrix(model, x_test, y_test, save_dir='data/plots'):
    """Plot and save the confusion matrix."""
    try:
        # Generate predictions
        y_pred = model.predict(x_test)
        y_pred_classes = y_pred.argmax(axis=1)
        y_true = y_test.argmax(axis=1)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        
        # Plot the confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save the plot
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, 'confusion_matrix.png')
        plt.savefig(plot_path)
        plt.show()
        logger.info(f"Confusion matrix saved at: {plot_path}")
    except Exception as e:
        logger.error("Error in plotting confusion matrix: %s", str(e))
        raise
