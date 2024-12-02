#Setting up EarlyStopping callback
early_stopping = EarlyStopping(
    #Monitors the validation loss during training
    monitor='val_loss', 
    #If validation loss doesn't improve for 3 consecutive epochs, training stops early
    #Change is accordingly to the number of epochs you want to wait before stopping
    patience=3,  
    #Restores the model's best weights (with the lowest validation loss) after stopping
    restore_best_weights=True,
    #Looks for the minimum value of 'val_loss' (we want to minimize the loss)
    mode='min'  
)

#Setting Bacth size fir training
batch_size = 16 

#Setting number of epochs for training
epochs = 10  

history = model.fit(x=train_gen,
                    epochs = epochs,
                    verbose = 1,
                    validation_data = validation_gen, 
                    validation_steps = None,
                    shuffle = False,
                    batch_size = batch_size,
                    callbacks = [early_stopping])