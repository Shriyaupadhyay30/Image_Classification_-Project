
#Defining image and batch size parameters

#Number of images to be processed in a batch
batch_size = 40  
#Cropping image size (width, height) in pixels
img_size = (224, 224)
#Number of color channels (RGB)
channels = 3

#Shape of the input image
img_shape = (img_size[0], img_size[1], channels)

#Calculating custom test batch size based on test dataset length
ts_length = len(test_df) 

#Finding the optimal test batch size where number of steps is <= 80
test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length % n == 0 and ts_length / n <= 80]))

#Calculating the number of steps per epoch for the test dataset
test_steps = ts_length // test_batch_size

#Custom scalar function to be used in the ImageDataGenerator; it returns the image without any changes
def scalar(img):
    return img

#Creating an ImageDataGenerator for training with data augmentation (rotation, shifting, zooming, flipping, etc.)
training_gen = ImageDataGenerator(preprocessing_function=scalar,  # Apply the scalar function to the images
                            #Data augmentation parameters
                            rotation_range=40,  
                            width_shift_range=0.2,  
                            height_shift_range=0.2,  
                            brightness_range=[0.4, 0.6],  
                            zoom_range=0.3, 
                            horizontal_flip=True, 
                            vertical_flip=True) 

#Creating a similar ImageDataGenerator for testing (no data augmentation, just scalar function)
#Appling the scalar function to the images
testing_gen = ImageDataGenerator(preprocessing_function=scalar,  
                            rotation_range=40,
                            width_shift_range=0.2,
                            height_shift_range=0.2,
                            brightness_range=[0.4, 0.6],
                            zoom_range=0.3,
                            horizontal_flip=True,
                            vertical_flip=True)

#Generating training data from a DataFrame
train_gen = training_gen.flow_from_dataframe(train_df,  #DataFrame with training data paths and labels
                                       #Column name for image file paths
                                       x_col='filepaths', 
                                       #Column name for image labels 
                                       y_col='labels',
                                       #Resize images to target size (224x224)  
                                       target_size=img_size,  
                                       #Class mode for categorical labels (multi-class classification)
                                       class_mode='categorical',
                                       #Load images as RGB (3 channels)  
                                       color_mode='rgb',
                                       #Shuffle the data for better training  
                                       shuffle=True,  
                                       #Number of images per batch
                                       batch_size=batch_size)  

#Generating validation data from a DataFrame
validation_gen = testing_gen.flow_from_dataframe(validation_df,  #DataFrame with validation data paths and labels
                                            x_col='filepaths',
                                            y_col='labels',
                                            target_size=img_size,
                                            class_mode='categorical',
                                            color_mode='rgb',
                                            shuffle=True,  #Shuffle validation data
                                            batch_size=batch_size)

#Generating test data from a DataFrame
#Using custom test_batch_size and no shuffling since the test data needs to be evaluated as it is
#DataFrame with test data paths and labels
test_gen = testing_gen.flow_from_dataframe(test_df,
                                      x_col='filepaths',
                                      y_col='labels',
                                      target_size=img_size,
                                      class_mode='categorical',
                                      color_mode='rgb',
                                      #Do not shuffle test data
                                      shuffle=False,  
                                      #Custom test batch size calculated earlier
                                      batch_size=test_batch_size) 
