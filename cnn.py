import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Paths to folders
train_folder = 'TrainIJCNN2013'

# Ground truth file path
gt_file = os.path.join(train_folder, 'gt.txt')
# Function to load training images and labels from the ground truth file
def load_train_data():
    images = []
    labels = []
    count = 0  # For debugging, to track image index

    with open(gt_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split(';')
            img_file, left, top, right, bottom, class_id = parts
            img_path = os.path.join(train_folder, img_file)
            
            # Read the image
            img = cv2.imread(img_path)
            
            # Ensure the image is loaded correctly
            if img is None:
                print(f"Error: Image {img_file} not found or could not be loaded.")
                continue
            
            # Convert the bounding box coordinates to integers
            left, top, right, bottom = map(int, [left, top, right, bottom])
            
            # Crop the ROI from the image
            cropped_img = img[top:bottom, left:right]
            
            try:
                # Resize the cropped image to 64x64
                resized_img = cv2.resize(cropped_img, (64, 64))
                images.append(resized_img)
                labels.append(int(class_id))
            except Exception as e:
                print(f"Error processing image {img_file} at index {count}: {e}")
            
            count += 1  # Increment index for debugging

    # Convert images and labels to numpy arrays
    try:
        images = np.array(images)
        labels = np.array(labels)
    except ValueError as ve:
        print(f"Error converting to numpy array: {ve}")
    
    # Normalize the images
    images = images / 255.0

    # Convert labels to one-hot encoding
    labels = to_categorical(labels, num_classes=43)  # 43 classes in total

    return images, labels

# Load the training data
X, Y = load_train_data()

# Split the data: 80% train+val, 20% test
X_train_val, X_test, Y_train_val, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Split train+val into 60% train and 20% val
X_train, X_val, Y_train, Y_val = train_test_split(X_train_val, Y_train_val, test_size=0.2, random_state=42)

print(f"Training data: {X_train.shape}, {Y_train.shape}")
print(f"Validation data: {X_val.shape}, {Y_val.shape}")
print(f"Test data: {X_test.shape}, {Y_test.shape}")

# Create the CNN model
def create_model():
    road_signs_model = Sequential()
    road_signs_model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    road_signs_model.add(MaxPooling2D((2, 2)))

    road_signs_model.add(Conv2D(64, (3, 3), activation='relu'))
    road_signs_model.add(MaxPooling2D((2, 2)))

    road_signs_model.add(Conv2D(128, (3, 3), activation='relu'))
    road_signs_model.add(MaxPooling2D((2, 2)))

    road_signs_model.add(Flatten())
    road_signs_model.add(Dense(128, activation='relu'))

    road_signs_model.add(Dropout(0.5))
    road_signs_model.add(Dense(43, activation='softmax'))  # 43 classes

    return road_signs_model

road_signs_model = create_model()
road_signs_model.summary()

# Data augmentation for training
augment_data = ImageDataGenerator(
    rotation_range=10,
    height_shift_range=0.1,
    width_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Prepare training generator
train_generator = augment_data.flow(X_train, Y_train, batch_size=32)

# Compile the model
road_signs_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping and model checkpoint callbacks
earlyStopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
modelCheckpoint = ModelCheckpoint('bestModel.keras', save_best_only=True, monitor='val_loss')

# Train the model
history = road_signs_model.fit(
    train_generator,
    validation_data=(X_val, Y_val),
    epochs=50,
    callbacks=[earlyStopping, modelCheckpoint]
)

# Evaluate on test data
test_loss, test_accuracy = road_signs_model.evaluate(X_test, Y_test, verbose=2)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Predictions on test data
y_pred = np.argmax(road_signs_model.predict(X_test), axis=-1)
y_true = np.argmax(Y_test, axis=-1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(43), yticklabels=range(43))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Ensure unique class check
unique_classes = np.unique(np.concatenate((y_true, y_pred)))
print(f"Unique classes in dataset: {unique_classes}")

# Generate classification report with adjusted target names
print(classification_report(
    y_true, y_pred, 
    target_names=[str(i) for i in unique_classes],
    labels=unique_classes
))

'''
Possible Reasons for Misclassification:

Similar Appearance: Road signs that look very similar (e.g., different speed limits with the same shape or color).
Insufficient Data: Some classes might have fewer samples in the training set, leading to poor generalization for those classes.
Poor Image Quality or Preprocessing Issues: Images with noise, blurriness, or poor lighting conditions can confuse the model. Data augmentation techniques like rotation, zooming, or changing brightness can help make the model more robust.
Class Imbalance: If certain classes are underrepresented in the training data, the model might have difficulty recognizing them. Techniques like class weighting, oversampling, or undersampling could be helpful here.
'''