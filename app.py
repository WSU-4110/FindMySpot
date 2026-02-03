from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # This allows your frontend to connect

@app.route('/')
def serve_frontend():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'index.html'), 'r') as f:
        return f.read()

@app.route('/api/hello', methods=['GET'])
def hello():
    return jsonify({'message': 'Backend is working!'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)   