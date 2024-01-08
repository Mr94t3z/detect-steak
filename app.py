from flask import Flask, render_template, request, url_for
from roboflow import Roboflow
import os
import tempfile
import cv2
import io
import base64
import supervision as sv
import uuid
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Define the upload folder

# Access the API key from environment variables
roboflow_api_key = os.getenv('ROBOFLOW_API_KEY')
rf = Roboflow(api_key=roboflow_api_key) if roboflow_api_key else None

project = rf.workspace().project("detect_steak")
model = project.version(1).model

# Create a variable to store the path of the uploaded file
uploaded_file_path = None


def get_file_extension(filename):
    return os.path.splitext(filename)[1]

@app.route('/', methods=['GET', 'POST'])
def index():
    global uploaded_file_path  # Access the global variable
    prediction = None
    annotated_image_base64 = None
    result = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_extension = get_file_extension(file.filename)
            random_filename = str(uuid.uuid4()) + file_extension
            temp_file = tempfile.NamedTemporaryFile(suffix=file_extension, delete=False)
            file.save(temp_file.name)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], random_filename)
            os.rename(temp_file.name, upload_path)
            
            # Save the path of the uploaded file
            uploaded_file_path = upload_path
            
            result = model.predict(upload_path, confidence=50, overlap=30).json()

            # Extract labels from Roboflow predictions
            labels = [item["class"] for item in result["predictions"]]

            # Convert Roboflow predictions to supervision detections
            detections = sv.Detections.from_roboflow(result)

            # Annotate the image using supervision library
            label_annotator = sv.LabelAnnotator()
            bounding_box_annotator = sv.BoxAnnotator()

            image = cv2.imread(upload_path)

            annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

            # Convert annotated_image from OpenCV to base64-encoded string
            _, buffer = cv2.imencode('.jpg', annotated_image)
            annotated_image_base64 = base64.b64encode(buffer).decode('utf-8')

            prediction = result

    return render_template('index.html', prediction_data=result, prediction=prediction, annotated_image_base64=annotated_image_base64)

@app.route('/delete', methods=['POST'])
def delete_uploaded_file():
    global uploaded_file_path  # Access the global variable
    if uploaded_file_path:
        os.remove(uploaded_file_path)  # Delete the uploaded file
        uploaded_file_path = None  # Reset the path variable
        return "File deleted"
    return "No file to delete"

@app.route('/clear_data', methods=['GET'])
def clear_data():
    # Clear all uploaded files in the UPLOAD_FOLDER
    folder = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

    # Additionally, reset any other data or variables you need to clear

    return "All data cleared"

if __name__ == '__main__':
    app.run(debug=True)
