from flask import Flask, request, render_template, jsonify
from app.inference import predict_image
from app.utils import setup_logging
import os
import traceback
import sys

app = Flask(__name__, static_folder='app/static', template_folder='app/templates')
setup_logging()

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        front_file = request.files.get("front")
        back_file = request.files.get("back")

        results = {}
        if front_file:
            label, _ = predict_image(front_file, side="front")
            results['front'] = label

        if back_file:
            label, _ = predict_image(back_file, side="back")
            results['back'] = label

        return render_template("upload.html", result=results)

    except Exception as e:
        print(f"🔥 Error in /predict: {e}")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        return render_template("upload.html", result={"error": "Internal Server Error"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)