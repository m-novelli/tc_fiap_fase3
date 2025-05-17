from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from logging.handlers import RotatingFileHandler
import os
from dotenv import load_dotenv
import joblib
import numpy as np
# Carregar variáveis de ambiente
load_dotenv(override=True)

# Inicializar aplicação Flask
app = Flask(__name__)
CORS(app)

# Configurar logging
if not os.path.exists('logs'):
    os.makedirs('logs')

file_handler = RotatingFileHandler('logs/ml_api.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [em %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)
app.logger.info('ML API started')

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint de verificação de saúde"""
    return jsonify({
        'status': 'ok',
        'mensagem': 'API running'
    }), 200

@app.route('/api/v1/predict', methods=['POST'])
def predict():
    """Endpoint de predição"""
    try:
        data = request.get_json()
        
        if not data or 'data' not in data:
            return jsonify({
                'erro': 'Formato de dados inválido. Esperado {"data": [sepal_length, sepal_width, petal_length, petal_width]}'
            }), 400
            
        app.logger.info(f"Dados de entrada para predição: {data}")
            
        model = joblib.load('model/rf_iris.joblib')
        features = np.array([data['data']])
        prediction = model.predict(features)
        
        # Converter array numpy para lista Python para serialização JSON
        prediction_list = prediction.tolist()
        
        # Mapear predições numéricas para nomes das classes
        class_names = ['setosa', 'versicolor', 'virginica']
        predicted_class = class_names[prediction_list[0]]
        
        return jsonify({
            'prediction': predicted_class,
            'prediction_numeric': prediction_list[0],
            'input_features': data['data']
        }), 200
        
    except Exception as e:
        app.logger.error(f'Error during prediction: {str(e)}')
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'error': 'Not found',
        'message': 'The requested resource was not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f'Server Error: {error}')
    return jsonify({
        'error': 'Internal server error',
        'message': 'An internal server error occurred'
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5150))
    app.run(host='0.0.0.0', port=port, debug=True)
