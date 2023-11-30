from flask import Flask, render_template, request, send_from_directory
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import openai
from creds import apikey
app = Flask(__name__)

# openai.api_key = apikey

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def process_image(image_path):
    # Load and preprocess the image for your model
    img = Image.open(image_path)
    img = img.resize((256, 256))  # adjust the size according to your model's input
    img_array = np.array(img)  # normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension
    return img_array

def predict_image_class(image_array, model, class_names):
    batch_prediction = model.predict(image_array)
    predicted_class_index = np.argmax(batch_prediction[0])
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        dropdown_data = request.form['dropdown_data']

        if 'file' not in request.files:
            return render_template('index.html', prediction='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', prediction='No selected file')

        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename) # type: ignore
            file.save(filename)

            # Display image in Flask application web page
            uploaded_image = file.filename

            dataset = tf.keras.preprocessing.image_dataset_from_directory(f"D:/B.Tech Semester 5/AI & ML Project/Dataset/{dropdown_data}", shuffle=True)
            class_names = dataset.class_names # type: ignore 

            # Process the uploaded image
            model = tf.keras.models.load_model(f"D:/B.Tech Semester 5/AI & ML Project/models/{dropdown_data}")
            img_array = process_image(filename)

            predicted_class_name = predict_image_class(img_array, model, class_names)

            # remedies = openai.Completion.create( # type: ignore
            # engine="text-davinci-002",
            # prompt=f"Suggest some medication methods for {predicted_class_name} plant leaf in paragraph only solution.",
            # max_tokens=150
            # )
            
            # symptoms = openai.Completion.create( # type: ignore
            # engine="text-davinci-002",
            # prompt=f"Symptoms of {predicted_class_name} plant leaf disease in paragraph.",
            # max_tokens=150
            # )
            # Extract the generated message
            # remedies_message = remedies['choices'][0]['text'].strip() # type: ignore
            # symptoms_message = symptoms['choices'][0]['text'].strip() # type: ignore

            # return render_template('index.html', prediction=predicted_class_name, uploaded_image=uploaded_image, remedies_gen=remedies_message, symptoms_gen=symptoms_message)
            return render_template('index.html', prediction=predicted_class_name, uploaded_image=uploaded_image)


    return render_template('index.html', prediction=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
