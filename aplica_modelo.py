

from datas.salva_base_localmente import buscar_dados_historicos
from models.ML_otimizacao_portfolio import PortfolioOptimizer
from datetime import datetime

tickers = ['AAPL','GOOG','AMZN', 'NFLX', 'MSFT', 'IBM','^GSPC']
tickers2 = ['AAPL','GOOG','AMZN', 'NFLX', 'MSFT', 'IBM']

start = "2007-01-01"
end = datetime.now().strftime("%Y-%m-%d")

df = buscar_dados_historicos(tickers, start, end, intervalo='1d')

optimizer = PortfolioOptimizer(df, tickers2, '^GSPC')

optimizer.load_data()

features_df, targets_dict = optimizer.prepare_ml_features()

models_dict = optimizer.read_joblib()

optimizer.optimize_ml_portfolio(models_dict, features_df)
