from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

#Loading the trained model
model = load_model('model_v2.h5')

#Defining classes from the dataset
class_labels = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

#Define the prediction and display function
def predict_and_display(img, model, class_labels):
    #Resize the image to the required input shape
    img = img.resize((224, 224))

    #Preprocess the image
    #image to array using keras
    img_array = keras_image.img_to_array(img)
    #Expanding dimensions to match model inpu
    img_array = np.expand_dims(img_array, axis=0) 
    #Preprocessing the image as required by EfficientNett
    img_array = preprocess_input(img_array)  

    #Making a prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)

    #Getting the class name from the defined list of class labels
    predicted_class_label = class_labels[predicted_class_index]

    #Getting the confidence percentage for the predicted class
    confidence_percentage = 100 * np.max(prediction)

    #Printing the class index for debugging
    print(f"Predicted Class Index: {predicted_class_index}")
    
    #Return prediction details
    #Converting the index to int for JSON
    return predicted_class_label, int(predicted_class_index), confidence_percentage


#Defining the API route for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    #handling missing file from API request
    if 'image' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    #Getting the file from the request
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    #Opening and processing the image
    try:
        img = Image.open(file.stream)
        predicted_class_label, predicted_class_index, confidence_percentage = predict_and_display(img, model, class_labels)

        #Returning the prediction as JSON
        return jsonify({
            'predicted_class': predicted_class_label,
            'predicted_class_index': predicted_class_index,
            'confidence_percentage': confidence_percentage
        })
    #Raising an exception if the image cannot be processed, with error code
    except Exception as e:
        return jsonify({"error": f"Error processing the image: {str(e)}"}), 500

#Defiing home route for API endpoint
@app.route('/')
def home():
    return "Welcome to the Image Classification API! Use the /predict endpoint to upload an image for prediction."

#running the api
if __name__ == '__main__':
    app.run(debug=True)
