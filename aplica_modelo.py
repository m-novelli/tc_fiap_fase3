from datas.salva_base_localmente import buscar_dados_historicos
from models.ML_otimizacao_portfolio import PortfolioOptimizer
from datetime import datetime
import pandas as pd

tickers = ['AAPL','GOOG','AMZN', 'NFLX', 'MSFT', 'IBM','^GSPC']

tickers_2 = ['AAPL','GOOG','AMZN', 'NFLX', 'MSFT', 'IBM']

start = "2007-01-01"

end = datetime.now().strftime("%Y-%m-%d")

df = buscar_dados_historicos(tickers, start, end, intervalo='1d')

#df = pd.read_csv('datas/dados_base.csv')

optimizer = PortfolioOptimizer(df, tickers_2, '^GSPC')

optimizer.load_data()

features_df, targets_dict = optimizer.prepare_ml_features()

models_dict = optimizer.read_joblib()

resultado = optimizer.optimize_ml_portfolio(models_dict, features_df)

optimizer.backtest_portfolio(weights_input=resultado['weights'], start_date_str=None, end_date_str=None)

optimizer.calculate_portfolio_performance(weights=list(resultado['weights'].values()))





