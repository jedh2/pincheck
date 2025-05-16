from flask import Flask, request, render_template, jsonify
from app.inference import predict_image
from app.utils import setup_logging
import os

app = Flask(__name__, static_folder='app/static', template_folder='app/templates')
setup_logging()

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    front_file = request.files.get('front')
    back_file = request.files.get('back')
    results = {}

    if front_file:
        results['front_prediction'], results['front_gradcam'] = predict_image(front_file, side='front')
    if back_file:
        results['back_prediction'], results['back_gradcam'] = predict_image(back_file, side='back')

    if not results:
        return jsonify({'error': 'No image uploaded'}), 400

    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)