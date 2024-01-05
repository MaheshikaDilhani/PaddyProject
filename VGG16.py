#!/usr/bin/env python
# coding: utf-8

# # Training Data set

# In[2]:


import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt


# In[2]:


IMAGE_SIZE=224
BATCH_SIZE=32
CHANNELS=3
EPOCHS=25


# In[3]:


dataset = tf.keras.preprocessing.image_dataset_from_directory(
   "D:\\Paddy_project\\paddyleaves_selected",
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)


# In[4]:


class_names=dataset.class_names
class_names


# In[5]:


len(dataset)


# In[6]:


50*32


# In[7]:


for image_batch, label_batch in dataset.take(1):
    print(image_batch.shape)
    print(label_batch.numpy())


# In[8]:


for image_batch, label_batch in dataset.take(1):
    for i in range(12):
        ax=plt.subplot(3,4,i+1)
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title(class_names[label_batch[i]])
        plt.axis("off")


# In[9]:


len(dataset)


# 80% ==>training
# 20%==>10% validation, 10% test

# In[10]:


train_size=0.8
len(dataset)*train_size


# In[11]:


train_ds=dataset.take(40)
len(train_ds)


# In[12]:


test_ds=dataset.skip(40)
len(test_ds)


# In[13]:


val_size=0.1
len(dataset)*val_size


# In[14]:


val_ds=test_ds.take(5)
len(val_ds)


# In[15]:


test_ds=test_ds.skip(5)
len(test_ds)


# In[16]:


def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    ds_size = len(ds)
    
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds


# In[17]:


train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)


# In[18]:


len(train_ds)


# In[19]:


len(val_ds)


# In[20]:


len(test_ds)


# Cache, Shuffle, and Prefetch the Dataset

# In[21]:


train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)


# In[22]:


resize_and_rescale = tf.keras.Sequential([
  layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
  layers.experimental.preprocessing.Rescaling(1./255),
])


# Data Augmentation

# In[23]:


data_augmentation = tf.keras.Sequential([
 layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
 layers.experimental.preprocessing.RandomRotation(0.2),
])


# Model Architecture

# In[24]:


input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 4

model = models.Sequential([
    resize_and_rescale,
    layers.Conv2D(32, kernel_size = (3,3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(n_classes, activation='softmax'),
])

model.build(input_shape=input_shape)


# In[25]:


model.summary()


# Compiling the Model

# In[28]:


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)


# In[29]:


history = model.fit(
    train_ds,
    batch_size=BATCH_SIZE,
    validation_data=val_ds,
    verbose=1,
    epochs=25,
)


# In[30]:


scores=model.evaluate(test_ds)


# In[31]:


scores


# Plotting the Accuracy and Loss Curves

# In[32]:


history


# In[33]:


history.params


# In[34]:


history.history.keys()


# loss, accuracy, val loss etc are a python list containing values of loss, accuracy etc at the end of each epoch

# In[35]:


type(history.history['loss'])


# In[36]:


len(history.history['loss'])


# In[37]:


history.history['loss'][:5] # show loss for first 5 epochs


# In[38]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']


# In[39]:


plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), acc, label='Training Accuracy')
plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), loss, label='Training Loss')
plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# Run prediction on a sample image

# In[40]:


import numpy as np
for images_batch, labels_batch in test_ds.take(1):
    
    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()
    
    print("first image to predict")
    plt.imshow(first_image)
    print("actual label:",class_names[first_label])
    
    batch_prediction = model.predict(images_batch)
    print("predicted label:",class_names[np.argmax(batch_prediction[0])])


# Write a function for inference

# In[41]:


def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence


# In[42]:


plt.figure(figsize=(15, 15))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        
        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i]] 
        
        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")
        
        plt.axis("off")


# In[43]:


model_version = 2
file_path = f"D:/Paddy_project/models/model_{model_version}"

model.save(file_path)


# In[55]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
# Evaluate the model on the test set
test_results = model.evaluate(test_ds)

# Make predictions on the test set
predictions = model.predict(test_ds)
y_true = []
y_pred = []

# Convert one-hot encoded predictions to class labels
for batch in test_ds:
    images, labels = batch
    y_true.extend(labels.numpy())
    y_pred.extend(tf.argmax(model.predict(images), axis=1).numpy())

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Classification Report
report = classification_report(y_true, y_pred, target_names=class_names)
print("Classification Report:")
print(report)

# Plot Confusion Matrix
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[56]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ... (your existing code)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Plot Confusion Matrix with Labels
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[57]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ... (your existing code)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Plot Confusion Matrix with Labels and Different Colormap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[58]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ... (your existing code)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Normalize Confusion Matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot Normalized Confusion Matrix with Labels and Different Colormap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# In[54]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# ... (your existing code)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Normalize Confusion Matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Plot Normalized Confusion Matrix with Labels and Different Colormap
plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, cmap="YlGnBu", xticklabels=class_names, yticklabels=class_names)
plt.title('Normalized Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()


# # VGG16

# In[59]:


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam

IMAGE_SIZE = 224
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 10

# Load the pre-trained VGG16 model (excluding the top layer)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))

# Freeze the convolutional layers of VGG16
for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top of VGG16
model = tf.keras.Sequential([
    layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
    layers.experimental.preprocessing.Rescaling(1./255),
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')  # Assuming 4 output classes
])

# Compile the model
model.compile(optimizer=Adam(lr=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Load your dataset
dataset_directory = "D:\\Paddy_project\\paddyleaves_selected"
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_directory,
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    validation_split=0.1,  # Adjust the validation split as needed
    subset="training"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_directory,
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    validation_split=0.1,  # Adjust the validation split as needed
    subset="validation"
)

# Continue with the rest of your code (data preprocessing, training, etc.)
# ...

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    verbose=1,
    epochs=EPOCHS,
)


# In[6]:


# Extract training and validation accuracy and loss from history
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS + 1), train_acc, label='Training Accuracy')
plt.plot(range(1, EPOCHS + 1), val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS + 1), train_loss, label='Training Loss')
plt.plot(range(1, EPOCHS + 1), val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[8]:


import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ... (Your existing code)

# Predict the labels for the validation dataset
y_val_pred = np.argmax(model.predict(val_ds), axis=-1)

# Extract true labels from the validation dataset
y_val_true = np.concatenate([y for x, y in val_ds], axis=0)

# Print and plot the classification report and confusion matrix
conf_matrix = confusion_matrix(y_val_true, y_val_pred)
print('Confusion Matrix:')
print(conf_matrix)

class_names = [str(i) for i in range(4)]  # Assuming 4 output classes

# Plot the confusion matrix using seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Print the classification report
class_report = classification_report(y_val_true, y_val_pred, target_names=class_names)
print('Classification Report:')
print(class_report)


# In[3]:


import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Assuming that you have a test dataset named test_ds
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_directory,
    seed=123,
    shuffle=False,  # Set shuffle to False for better evaluation metrics
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_ds)

print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Predict the labels for the test dataset
y_pred = np.argmax(model.predict(test_ds), axis=-1)

# Extract true labels from the test dataset
y_true = np.concatenate([y for x, y in test_ds], axis=0)

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print('Confusion Matrix:')
print(conf_matrix)

# Calculate classification report (includes precision, recall, and F1 score)
class_report = classification_report(y_true, y_pred)
print('Classification Report:')
print(class_report)


# In[5]:


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

IMAGE_SIZE = 224
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 5

# Load the pre-trained VGG16 model (excluding the top layer)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))

# Freeze the convolutional layers of VGG16
for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top of VGG16
model = tf.keras.Sequential([
    layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
    layers.experimental.preprocessing.Rescaling(1./255),
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')  # Assuming 4 output classes
])

# Compile the model
model.compile(optimizer=Adam(lr=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Load your dataset
dataset_directory = "D:\\Paddy_project\\paddyleaves_selected"
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_directory,
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    validation_split=0.1,  # Adjust the validation split as needed
    subset="training"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_directory,
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    validation_split=0.1,  # Adjust the validation split as needed
    subset="validation"
)

# Continue with the rest of your code (data preprocessing, training, etc.)
# ...

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    verbose=1,
    epochs=EPOCHS,
)

# Extract training and validation accuracy and loss from history
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# Plot training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS + 1), train_acc, label='Training Accuracy')
plt.plot(range(1, EPOCHS + 1), val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS + 1), train_loss, label='Training Loss')
plt.plot(range(1, EPOCHS + 1), val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[7]:


import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Get the final validation accuracy
final_val_accuracy = val_acc[-1]
print(f'Final Validation Accuracy: {final_val_accuracy * 100:.2f}%')

# Evaluate the model on the validation dataset
val_loss, val_accuracy = model.evaluate(val_ds)
print(f'Validation Loss: {val_loss:.4f}')
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

# Predict the labels for the validation dataset
y_val_pred = np.argmax(model.predict(val_ds), axis=-1)

# Extract true labels from the validation dataset
y_val_true = np.concatenate([y for x, y in val_ds], axis=0)

# Calculate precision, recall, and F1 score
precision = precision_score(y_val_true, y_val_pred, average='weighted')
recall = recall_score(y_val_true, y_val_pred, average='weighted')
f1 = f1_score(y_val_true, y_val_pred, average='weighted')

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')


# In[2]:


import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = 224
BATCH_SIZE = 32
CHANNELS = 3
EPOCHS = 25

# Load the pre-trained VGG16 model (excluding the top layer)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS))

# Freeze the convolutional layers of VGG16
for layer in base_model.layers:
    layer.trainable = False

# Create a new model on top of VGG16
model = tf.keras.Sequential([
    layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
    layers.experimental.preprocessing.Rescaling(1./255),
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')  # Assuming 4 output classes
])

# Compile the model
model.compile(optimizer=Adam(lr=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print model summary
model.summary()

# Data Augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Load your dataset
dataset_directory = "D:\\Paddy_project\\paddyleaves_selected"
train_datagen = datagen.flow_from_directory(
    dataset_directory,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='training'
)

val_datagen = datagen.flow_from_directory(
    dataset_directory,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    subset='validation'
)

# Train the model
history = model.fit(
    train_datagen,
    validation_data=val_datagen,
    verbose=1,
    epochs=EPOCHS,
)


# In[ ]:




