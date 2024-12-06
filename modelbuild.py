#Imports
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from keras.callbacks import EarlyStopping
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image


class MangoLeafDiseaseModel:
    """
    A class to model the Mango Leaf Disease classification problem.
    It handles the data preparation, model building, training, evaluation, and saving the model.
    """
    def __init__(self, data_directory, img_size=(224, 224), batch_size=40):
        """
        Initializes the model with the dataset directory, image size, and batch size.
        """
        #Path to the data
        self.data_directory = data_directory  
        #Size of input images for the model
        self.img_size = img_size  
        #Number of images per batch
        self.batch_size = batch_size  
        self.train_df = None
        self.validation_df = None
        self.test_df = None
        self.model = None

    def get_data_paths(self):
        """
        Gets the file paths and labels for the dataset. Assumes that the data is organized by classes in subdirectories.
        """
        filepaths = []
        labels = []

        #Getting the class folders from the directory
        folds = os.listdir(self.data_directory)

        for fold in folds:
            #Path for each class folder
            foldpath = os.path.join(self.data_directory, fold) 
            #Listing of files (images) in the class folder 
            filelist = os.listdir(foldpath)  
            for file in filelist:
                #Full path to each image file
                fpath = os.path.join(foldpath, file)  
                filepaths.append(fpath)
                #Assigning label based on folder name (class)
                labels.append(fold)  

        return filepaths, labels

    def create_df(self, filepaths, labels):
        """
        Creates a pandas dataframe from the file paths and labels.
        """
        #Series for file paths
        Fseries = pd.Series(filepaths, name='filepaths') 
        #Series for corresponding labels 
        Lseries = pd.Series(labels, name='labels') 
        #Combine into a single dataframe 
        return pd.concat([Fseries, Lseries], axis=1)  

    def data_cleaning(self, df):
        """
        Cleans the dataframe by removing rows with missing or duplicate values.
        """
        #Checking for null values
        num_null_vals = sum(df.isnull().sum().values)  
        if num_null_vals == 0:
            print("No null values.")
        else:
            print(f"There are {num_null_vals} null values. Removing rows with nulls.")
            #Dropping rows with missing values
            df = df.dropna()  

        #Checking for duplicate rows 
        num_duplicates = df.duplicated().sum()  
        if num_duplicates == 0:
            print("No duplicate values.")
        else:
            print(f"There are {num_duplicates} duplicate rows. Removing duplicates.")
            #Removing duplicate rows
            df = df.drop_duplicates()  

        return df

    def split_data(self, df):
        """
        Splits the data into training (70%), validation (15%), and test (15%) sets.
        """
        train_df, dummy_df = train_test_split(df, train_size=0.7, shuffle=True, random_state=123)
        validation_df, test_df = train_test_split(dummy_df, train_size=0.5, shuffle=True, random_state=123)
        return train_df, validation_df, test_df

    def create_generators(self, train_df, validation_df, test_df):
        """
        Creates image data generators for training, validation, and test sets.
        """
        def scalar(img):
            return img  #No scaling function applied

        #Data augmentation and preprocessing for training images
        training_gen = ImageDataGenerator(
            preprocessing_function=scalar,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            brightness_range=[0.4, 0.6],
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True
        )

        #Similar preprocessing for validation and test sets (without augmentation)
        testing_gen = ImageDataGenerator(
            preprocessing_function=scalar,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            brightness_range=[0.4, 0.6],
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True
        )

        #Creating training data generator
        train_gen = training_gen.flow_from_dataframe(
            train_df, x_col='filepaths', y_col='labels', target_size=self.img_size,
            class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=self.batch_size
        )

        #Creating validation data generator
        validation_gen = testing_gen.flow_from_dataframe(
            validation_df, x_col='filepaths', y_col='labels', target_size=self.img_size,
            class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=self.batch_size
        )

        #Creating test data generator
        ts_length = len(test_df)
        test_batch_size = max([ts_length // n for n in range(1, ts_length + 1) if ts_length % n == 0 and ts_length / n <= 80])
        test_gen = testing_gen.flow_from_dataframe(
            test_df, x_col='filepaths', y_col='labels', target_size=self.img_size,
            class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=test_batch_size
        )

        return train_gen, validation_gen, test_gen

    def build_model(self):
        """
        Builds the model using EfficientNetB0 as the base and adding custom layers on top.
        """
        #Input shape for the images
        img_shape = (self.img_size[0], self.img_size[1], 3)  
        #Number of output classes
        class_count = len(list(self.train_gen.class_indices.keys()))  

        #Loading EfficientNetB0 base model without the top layers
        base_model = EfficientNetB0(include_top=False, weights="imagenet", input_shape=img_shape, pooling='max')
        #Freezing the base model's layers
        base_model.trainable = False  

        #Building the final model with additional layers
        self.model = Sequential([
            #Adding the base model (EfficientNetB0)
            base_model,  
            #Batch Normalization
            BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001), 
            #Dense layer with L2 regularization 
            Dense(128, kernel_regularizer=regularizers.l2(0.01), activation='relu'),  
            #Dropout layer to prevent overfitting
            Dropout(rate=0.3),  
            #Final output layer with softmax activation for classification
            Dense(class_count, activation='softmax')  
        ])

        #Compiling the model
        self.model.compile(optimizer=Adamax(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    def train_model(self, epochs=100):
        """
        Trains the model with the provided data generators for a specified number of epochs.
        """
        #Implementing early stop with validation loss to stop the training if the model is not improving
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True, mode='min'
        )

        #Fitting the model on the training data
        self.history = self.model.fit(
            x=self.train_gen, epochs=epochs, validation_data=self.validation_gen,
            shuffle=False, batch_size=self.batch_size, callbacks=[early_stopping]
        )

    def evaluate_model(self):
        """
        Evaluates the model on the test data and prints the classification report.
        """
        #Getting predictions on the test set
        preds = self.model.predict(self.test_gen) 
        #Getting the class with the highest probability for each prediction 
        y_pred = np.argmax(preds, axis=1)  

        #Getting the class labels from the test generator
        g_dict = self.test_gen.class_indices
        classes = list(g_dict.keys())

        #Printing the classification report
        print(classification_report(self.test_gen.classes, y_pred, target_names=classes))

    def save_model(self, model_filename='mango_leaf_disease_model.h5'):
        """
        Saves the trained model to a file.
        """
        #Saving the trained model to use it later
        self.model.save(model_filename)
        print(f"Model saved to {model_filename}")


if __name__ == '__main__':
    #Dataset location
    data_directory = 'D:/Big Data Analytics/Term-2/BDM 3014 - Introduction to Artificial Intelligence 01/Final Project/MangoLeafBD Dataset'

    #Initializing the model
    model = MangoLeafDiseaseModel(data_directory)

    #Getting file paths and labels
    filepaths, labels = model.get_data_paths()

    #Creating dataframe
    df = model.create_df(filepaths, labels)

    #Cleaning the data (remove nulls and duplicates)
    df = model.data_cleaning(df)

    #Spliting data into training, validation, and test sets
    model.train_df, model.validation_df, model.test_df = model.split_data(df)

    #Creating data generators for the training, validation, and test sets
    model.train_gen, model.validation_gen, model.test_gen = model.create_generators(model.train_df, model.validation_df, model.test_df)

    #Building the model
    model.build_model()

    #Training the model
    model.train_model(epochs=100)

    #Evaluating the model
    model.evaluate_model()

    #Saveing the trained model
    model.save_model()
