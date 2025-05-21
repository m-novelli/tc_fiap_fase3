from datas.salva_base_localmente import buscar_dados_historicos
from models.ML_otimizacao_portfolio import PortfolioOptimizer
from datetime import datetime

tickers = ['AAPL','GOOG','AMZN', 'NFLX', 'MSFT', 'IBM','^GSPC']
start = "2007-01-01"
end = datetime.now().strftime("%Y-%m-%d")

df = buscar_dados_historicos(tickers, start, end, intervalo='1d')

optimizer = PortfolioOptimizer(df, tickers, '^GSPC')

optimizer.load_data()

optimizer.prepare_ml_features()

models_dict = optimizer.read_joblib()

optimizer.optimize_ml_portfolio(models_dict, optimizer.features_df)
