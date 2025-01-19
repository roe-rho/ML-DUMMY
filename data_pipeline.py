from functions import *
import time
import datetime

# Starting the pipeline
print("Starting data pipeline at ", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
print("-----------------------------------------------")

# Step 1: Extract data
t0 = time.time()
x_train, y_train, x_test, y_test = extract_data()
t1 = time.time()
print("Step 1: Data extracted")
print("---> Data extraction completed in", str(t1-t0), "seconds\n")

# Step 2: Create CNN model
t0 = time.time()
model = create_cnn()
t1 = time.time()
print("Step 2: CNN model created")
print("---> Model creation completed in", str(t1-t0), "seconds\n")

# Step 3: Train the model
t0 = time.time()
history = train_model(model, x_train, y_train)
t1 = time.time()
print("Step 3: Model trained")
print("---> Model training completed in", str(t1-t0), "seconds\n")

# Step 4: Save the model
t0 = time.time()
save_model(model, 'data/model/cnn_cifar10.h5')
t1 = time.time()
print("Step 4: Model saved")
print("---> Model saving completed in", str(t1-t0), "seconds\n")

# Step 5: Visualize training results
t0 = time.time()
plot_training(history.history, 'data/plots')
t1 = time.time()
print("Step 5: Training visualization completed")
print("---> Visualization completed in", str(t1-t0), "seconds\n")

# Step 6: Generate confusion matrix
t0 = time.time()
plot_confusion_matrix(model, x_test, y_test, 'data/plots')
t1 = time.time()
print("Step 6: Confusion matrix generated")
print("---> Confusion matrix generation completed in", str(t1-t0), "seconds\n")

# Step 7: Evaluate the model
t0 = time.time()
evaluate_model(model, x_test, y_test)
t1 = time.time()
print("Step 7: Model evaluation completed")
print("---> Evaluation completed in", str(t1-t0), "seconds\n")

print("----------------------------------------------")
print("Data pipeline completed at ", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
