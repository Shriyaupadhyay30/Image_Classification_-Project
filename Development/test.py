from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Load the trained model (adjust the path to where your model is saved)
model = load_model('model_v2.h5')

# Manually define your class labels (in order)
class_labels = ['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back', 'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']

# Define the prediction and display function
def predict_and_display(img, model, class_labels):
    # Resize the image to the required input shape of EfficientNetB0
    img = img.resize((224, 224))

    # Preprocess the image
    img_array = keras_image.img_to_array(img)  # Use keras_image to avoid conflict with PIL
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match model input
    img_array = preprocess_input(img_array)  # Preprocess the image as required by EfficientNet

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)

    # Get the class name from the manually defined list of class labels
    predicted_class_label = class_labels[predicted_class_index]

    # Get the confidence percentage for the predicted class
    confidence_percentage = 100 * np.max(prediction)

    # Return prediction details
    return predicted_class_label, confidence_percentage


# Define the API route for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    # Get the file from the request
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Open and process the image
    try:
        img = Image.open(file.stream)
        predicted_class_label, confidence_percentage = predict_and_display(img, model, class_labels)

        # Return the prediction as JSON
        return jsonify({
            'predicted_class': predicted_class_label,
            'confidence_percentage': confidence_percentage
        })

    except Exception as e:
        return jsonify({"error": f"Error processing the image: {str(e)}"}), 500


@app.route('/')
def home():
    return "Welcome to the Image Classification API! Use the /predict endpoint to upload an image for prediction."

if __name__ == '__main__':
    app.run(debug=True)
