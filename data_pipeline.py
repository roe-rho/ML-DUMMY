from functions import *
import time
import datetime
import logging  # Import logging

# Configure logging
logging.basicConfig(
    filename='training.log',  # Log file name
    level=logging.INFO,       # Logging level
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()  # Create logger instance

# Starting the pipeline
logger.info("Starting data pipeline at %s", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
logger.info("-----------------------------------------------")

try:
    # Step 1: Extract data
    t0 = time.time()
    x_train, y_train, x_test, y_test = extract_data()
    t1 = time.time()
    logger.info("Step 1: Data extracted")
    logger.info("---> Data extraction completed in %s seconds", str(t1 - t0))

    # Step 2: Create CNN model
    t0 = time.time()
    model = create_cnn()
    t1 = time.time()
    logger.info("Step 2: CNN model created")
    logger.info("---> Model creation completed in %s seconds", str(t1 - t0))

    # Step 3: Train the model
    t0 = time.time()
    history = train_model(model, x_train, y_train)
    t1 = time.time()
    logger.info("Step 3: Model trained")
    logger.info("---> Model training completed in %s seconds", str(t1 - t0))

    # Step 4: Save the model summary
    t0 = time.time()
    save_model_summary(model, 'data/model_summary.txt')
    t1 = time.time()
    logger.info("Step 4: Model summary saved")
    logger.info("---> Model summary saving completed in %s seconds", str(t1 - t0))

    # Step 5: Save the model
    t0 = time.time()
    save_model(model, 'data/model/cnn_cifar10.h5')
    t1 = time.time()
    logger.info("Step 5: Model saved")
    logger.info("---> Model saving completed in %s seconds", str(t1 - t0))

    # Step 6: Visualize training results
    t0 = time.time()
    plot_training(history.history, 'data/plots')
    t1 = time.time()
    logger.info("Step 6: Training visualization completed")
    logger.info("---> Visualization completed in %s seconds", str(t1 - t0))

    # Step 7: Generate confusion matrix
    t0 = time.time()
    plot_confusion_matrix(model, x_test, y_test, 'data/plots')
    t1 = time.time()
    logger.info("Step 7: Confusion matrix generated")
    logger.info("---> Confusion matrix generation completed in %s seconds", str(t1 - t0))

    # Step 8: Evaluate the model
    t0 = time.time()
    evaluate_model(model, x_test, y_test)
    t1 = time.time()
    logger.info("Step 8: Model evaluation completed")
    logger.info("---> Evaluation completed in %s seconds", str(t1 - t0))

except Exception as e:
    logger.error("An error occurred during the pipeline execution: %s", str(e))

finally:
    logger.info("----------------------------------------------")
    logger.info("Data pipeline completed at %s", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
