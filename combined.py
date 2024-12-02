# Imports
import os
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import random
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.offline import iplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import missingno as msno
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, EfficientNetB7

# Setting up dataset directory and name
data_directory = 'D:/Big Data Analytics/Term-2/BDM 3014 - Introduction to Artificial Intelligence 01/Final Project/MangoLeafBD Dataset'
dataset_name = 'Mango Leaf Disease Dataset'

# Function to get data paths and labels
def get_data_paths(data_directory):
    filepaths = []
    labels = []
    folds = os.listdir(data_directory)
    for fold in folds:
        foldpath = os.path.join(data_directory, fold)
        filelist = os.listdir(foldpath)
        for file in filelist:
            fpath = os.path.join(foldpath, file)
            filepaths.append(fpath)
            labels.append(fold)
    return filepaths, labels

filepaths, labels = get_data_paths(data_directory)

# Create dataframe with file paths and labels
def create_df(filepaths, labels):
    Fseries = pd.Series(filepaths, name='filepaths')
    Lseries = pd.Series(labels, name='labels')
    df = pd.concat([Fseries, Lseries], axis=1)
    return df

df = create_df(filepaths, labels)

# Data cleaning function
def data_cleaning(df, name='df'):
    num_null_vals = sum(df.isnull().sum().values)
    if num_null_vals == 0:
        print(f"The {name} has no null values.")
    else:
        print(f"The {name} has {num_null_vals} null values.")
        df = df.dropna()
    
    num_duplicates = df.duplicated().sum()
    if num_duplicates == 0:
        print(f"The {name} has no duplicate values.")
    else:
        print(f"The {name} has {num_duplicates} duplicate rows.")
        df = df.drop_duplicates()
    return df

df = data_cleaning(df, dataset_name)

# Class distribution visualization
def class_distribution(dataframe, col_name):
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Percentage Plot', 'Total Count Plot'))
    total_count = dataframe[col_name].value_counts().sum()
    percentage_values = (dataframe[col_name].value_counts().values / total_count) * 100
    fig.add_trace(go.Bar(y=percentage_values.tolist(),
                         x=[str(i) for i in dataframe[col_name].value_counts().index],
                         text=[f'{val:.2f}%' for val in percentage_values], 
                         textposition='auto'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=dataframe[col_name].value_counts().keys(),
                             y=dataframe[col_name].value_counts().values,
                             mode='markers'),
                  row=1, col=2)
    fig.update_layout(title={'text': 'Disease Distribution in Dataset'})
    iplot(fig)

class_distribution(df, 'labels')

# Train-test-validation split
train_df, dummy_df = train_test_split(df, train_size=0.7, shuffle=True, random_state=123)
validation_df, test_df = train_test_split(dummy_df, train_size=0.5, shuffle=True, random_state=123)

# Sequential model building
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)
class_count = len(list(train_df['labels'].unique()))

base_model = EfficientNetB7(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')
base_model.trainable = False

model = Sequential([
    base_model,
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
    Dense(128, kernel_regularizer=regularizers.l2(0.016), activity_regularizer=regularizers.l1(0.006),
          bias_regularizer=regularizers.l1(0.006), activation='relu'),
    Dropout(rate=0.45, seed=123),
    Dense(class_count, activation='softmax')
])

model.compile(optimizer=Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, mode='min')

# Training the model
epochs = 10
history = model.fit(x=train_df,
                    epochs=epochs,
                    validation_data=validation_df,
                    callbacks=[early_stopping])

# Evaluation
train_score = model.evaluate(train_df)
val_score = model.evaluate(validation_df)
test_score = model.evaluate(test_df)

print("Train Loss: ", train_score[0], "Train Accuracy: ", train_score[1])
print("Validation Loss: ", val_score[0], "Validation Accuracy: ", val_score[1])
print("Test Loss: ", test_score[0], "Test Accuracy: ", test_score[1])

# Confusion matrix
preds = model.predict(test_df)
y_pred = np.argmax(preds, axis=1)
cm = confusion_matrix(test_df['labels'], y_pred)

plt.figure(figsize=(10, 10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.show()
