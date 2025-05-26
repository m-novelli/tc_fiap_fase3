from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from logging.handlers import RotatingFileHandler
import os
from dotenv import load_dotenv
import joblib
import numpy as np
from models.PortfolioOptimizer import PortfolioOptimizer
from datas.salva_base_localmente import executar_pipeline_local

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
        benchmark_ticker = data['benchmark_ticker']
        user_tickers = data['user_tickers']
        download_tickers = data['user_tickers'].copy()
        download_tickers.append(benchmark_ticker)
        
        start_date = data['start_date']
        end_date = data['end_date']
        caminho = 'datas/dados_base_predict.csv'
        
        app.logger.info(f'Downloading data from {start_date} to {end_date} for tickers {user_tickers}: {caminho}')
        executar_pipeline_local(download_tickers, start_date, end_date, caminho)
        
        app.logger.info('Initializing optimizer...')
        optimizer = PortfolioOptimizer(caminho, user_tickers, benchmark_ticker)
        
        app.logger.info('Reading models...')
        models_dict = optimizer.read_joblib()
        models_to_use = {k: v for k, v in models_dict.items() if k in user_tickers}
        app.logger.info(f'Models to use: {models_to_use}')
        
        try:
            app.logger.info('Preparing features...')
            features_df, targets_dict  = optimizer.prepare_ml_features(window_size=30, target_window=10)
        except Exception as e:
            app.logger.error(f'Error during features preparation: {str(e)}')
            return jsonify({
                'error': 'Internal server error',
                'message': str(e)
            }), 500
        
        try:
            app.logger.info('Optimizing portfolio...')
            ml_optimization_results = optimizer.optimize_ml_portfolio(models_to_use, features_df)
        except Exception as e:
            app.logger.error(f'Error during ML optimization: {str(e)}')
            return jsonify({
                'error': 'Internal server error',
                'message': str(e)
            }), 500
            
        return jsonify({
            'prediction': ml_optimization_results
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
    
def load_models():
    return True

if __name__ == '__main__':
    load_models()
    port = int(os.environ.get('PORT', 5150))
    app.run(host='0.0.0.0', port=port, debug=True)
