from flask import Flask, request, jsonify
from app.inference import predict_class, refresh_model

app = Flask(__name__)

@app.route('/index')
def index():
    return 'MNIST inference deployed on K8S\n'

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        # Make a prediction using the inference module
        result = predict_class(file)
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/refresh')
def refresh():
    refresh_model()
    return 'Model refreshed successfully\n'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
