#Creating Model Structure
img_size = (224, 224)
channels = 3
img_shape = (img_size[0], img_size[1], channels)

#to define number of classes in dense layer
class_count = len(list(train_gen.class_indices.keys())) 


#using efficientnetb0 from EfficientNet family.
#Simpler base model istead of complex one, 
#because small dataset and to stop overfitting

base_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top= False, weights= "imagenet", input_shape= img_shape, pooling= 'max')
base_model.trainable = False

#Building a Sequential model with the EfficientNetB7 base
model = Sequential([
    #Adding the base model
    base_model,  

    #Normalizing inputs for faster training and convergence
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001), 
    #Adding a fully connected layer with 128 units 
    Dense(128,  
          #Adding L2 regularization to the weights
          kernel_regularizer=regularizers.l2(0.01), 
          #Adding L1 regularization to the activations 
          activity_regularizer=regularizers.l1(0.001), 
          #Adding L1 regularization to the biases 
          bias_regularizer=regularizers.l1(0.001),  
          #Using ReLU activation function
          activation='relu'), 
    #Dropout layer to prevent overfitting with a dropout rate of 45% 
    Dropout(rate=0.3, seed=123), 

    #Output layer with softmax activation for multi-class classification 
    Dense(class_count, activation='softmax')  
])

#Compiling the model
#Adamax is an adaptive learning rate optimizer based on Adam
#categorical_crossentropy is used as the loss function for multi-class classification
#Using Adamax optimizer with learning rate 0.001
model.compile(optimizer=Adamax(learning_rate=0.0001),  
               #Loss function for categorical classification
              loss='categorical_crossentropy', 
              #Metric to monitor during training is accuracy
              metrics=['accuracy'])  

#Displaying the model architecture summary
model.summary()