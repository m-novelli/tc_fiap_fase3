"""
Script para Otimização de Portfólio de Investimentos

Este script implementa e demonstra duas abordagens principais para a otimização de portfólio:
1.  Teoria Moderna do Portfólio (Markowitz): Busca maximizar o Índice de Sharpe, encontrando a alocação ótima de ativos que oferece o maior retorno para um dado nível de risco (ou o menor risco para um dado retorno).
2.  Otimização com Machine Learning: Utiliza modelos de Machine Learning (Random Forest Regressor) para prever os retornos futuros dos ativos e, com base nessas previsões, determinar os pesos do portfólio.


Fundamentação Teórica Breve:

Otimização com Machine Learning:
-   Abordagem: Utiliza modelos de ML para prever características futuras dos ativos (ex: retornos) e usa essas previsões para informar a alocação de ativos.
-   Vantagens Potenciais: Pode capturar relações não lineares e padrões complexos que modelos tradicionais podem não identificar. Pode adaptar-se a mudanças nas condições de mercado se treinado e atualizado regularmente.
-   Desafios: Prever mercados financeiros é inerentemente difícil. Modelos de ML podem sofrer de overfitting (bom desempenho em dados de treino, ruim em dados novos) e requerem validação cuidadosa (ex: backtesting robusto).
-   Neste script: Usamos Random Forest Regressor para prever retornos futuros de 30 dias para cada ativo, com base em features como retornos passados, volatilidade e momentum. Os pesos do portfólio são então derivados dessas previsões.

Backtesting:
-   Processo de testar uma estratégia de investimento usando dados históricos para ver como ela teria se saído no passado.
-   Crucial para avaliar a viabilidade de uma estratégia antes de implementá-la com capital real.
-   Métricas Comuns: Retorno Total, Retorno Anualizado, Volatilidade Anualizada, Índice de Sharpe, Máximo Drawdown (maior queda percentual do pico ao vale).
"""

# Importação das bibliotecas padrão e de terceiros necessárias para o script.
import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize 
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import datetime 
import json
import joblib
import os

# --- Configurações Globais --- 
# Usada no cálculo do Índice de Sharpe. Um valor comum é 2% (0.02).
RISK_FREE_RATE = 0.02

class PortfolioOptimizer:
    """
    Classe principal para encapsular as funcionalidades de otimização de portfólio.
    Permite carregar dados, realizar a otimização de Markowitz, treinar modelos de ML,
    otimizar com base em ML e realizar backtesting das estratégias.
    """

    def __init__(self, df_total_data, tickers_list, benchmark_ticker, risk_free_rate=RISK_FREE_RATE):
        """
        Inicializa o otimizador de portfólio.

        Args:
            df_total_data (pd.DataFrame): DataFrame contendo os dados históricos dos ativos e do benchmark
            tickers_list (list): Lista de strings contendo os tickers (símbolos) dos ativos
                                 que farão parte do portfólio a ser otimizado.
            benchmark_ticker (str): String do ticker do ativo que será usado como benchmark
                                    (ex: '^GSPC' para o S&P 500).
            risk_free_rate (float, optional): A taxa livre de risco anual. Padrão é o valor global RISK_FREE_RATE.
        """
        # Armazena os dados de entrada e parâmetros
        self.df_total = df_total_data
        self.tickers_list = tickers_list
        self.benchmark_ticker = benchmark_ticker
        self.risk_free_rate = risk_free_rate

        # Atributos que serão populados após o carregamento e processamento dos dados
        self.returns_data = None      # DataFrame com os retornos diários dos ativos do portfólio.
        self.prices_data = None       # DataFrame com os preços de fechamento dos ativos do portfólio.
        self.benchmark_returns = None # Series com os retornos diários do benchmark.

        # Chama o método para carregar e processar os dados assim que a classe é instanciada.
        self.load_data()

    def load_data(self):
        """
        Carrega e processa os dados do DataFrame `df_total` fornecido na inicialização.
        Este método realiza as seguintes etapas:
        1.  Cria uma cópia do DataFrame original para evitar modificações indesejadas.
        2.  Verifica se as colunas necessárias ('Date', 'Close', 'ticker') existem.
        3.  Converte a coluna 'Date' para o formato datetime.
        4.  Filtra o DataFrame para manter apenas os tickers relevantes (ativos do portfólio + benchmark).
        5.  Transforma o DataFrame do formato 'long' 
        6.  Verifica se todos os tickers esperados (ativos e benchmark) estão presentes após o pivot.
        7.  Preenche valores ausentes (NaN) usando forward fill (`ffill`) e depois backward fill (`bfill`).
        8.  Calcula os retornos percentuais diários (`pct_change`) a partir dos preços de fechamento.
        9.  Remove quaisquer linhas com NaN resultantes do cálculo de `pct_change` (geralmente a primeira linha).
        10. Separa os retornos do benchmark dos retornos dos ativos do portfólio.
        11. Armazena os DataFrames de retornos e preços processados nos atributos da classe.
        12. Imprime informações sobre os dados carregados (período, número de dias, tickers).

        Raises:
            ValueError: Se colunas obrigatórias estiverem faltando no `df_total`,
                        se o DataFrame ficar vazio após a filtragem por tickers,
                        se o benchmark ou algum dos tickers do portfólio não for encontrado nos dados,
                        ou se os DataFrames de preços ou retornos ficarem vazios após o processamento.
        """
        print("Carregando e processando dados do DataFrame fornecido...")
        df = self.df_total.copy() 

        # Verificação das colunas obrigatórias
        required_cols = ['Date', 'Close', 'ticker']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"O DataFrame `df_total` deve conter as colunas: {', '.join(required_cols)}")

        # Conversão da coluna de data e filtragem por tickers relevantes
        df['Date'] = pd.to_datetime(df['Date'])
        relevant_tickers = self.tickers_list + [self.benchmark_ticker]
        df = df[df['ticker'].isin(relevant_tickers)]

        if df.empty:
            raise ValueError("O DataFrame ficou vazio após filtrar pelos tickers fornecidos. Verifique os nomes dos tickers e se há dados para eles no `df_total`.")

        # Pivotar o DataFrame para ter datas como índice e tickers como colunas de preços ('Close')
        prices_wide = df.pivot(index='Date', columns='ticker', values='Close')
        
        # Verificar se o benchmark e os tickers do portfólio estão presentes
        if self.benchmark_ticker not in prices_wide.columns:
            raise ValueError(f"Ticker do benchmark '{self.benchmark_ticker}' não encontrado nos dados após pivotar. Verifique se há dados para este ticker no `df_total`.")
        missing_tickers = [ticker for ticker in self.tickers_list if ticker not in prices_wide.columns]
        if missing_tickers:
            raise ValueError(f"Os seguintes tickers do portfólio não foram encontrados nos dados após pivotar: {', '.join(missing_tickers)}. Verifique se há dados para eles.")

        # Preencher valores ausentes (NaN) que podem surgir de dias não negociados para alguns ativos.
        # ffill preenche com o último valor válido; bfill preenche com o próximo valor válido.
        prices_wide = prices_wide.ffill().bfill()
        
        if prices_wide.empty or prices_wide.isnull().all().all():
             raise ValueError("O DataFrame de preços (prices_wide) está vazio ou contém apenas NaNs após o preenchimento. Verifique a qualidade dos dados de entrada.")

        # Calcular os retornos diários (variação percentual do preço de um dia para o outro)
        returns_wide = prices_wide.pct_change().dropna() # dropna() remove a primeira linha que será NaN
        
        if returns_wide.empty:
            raise ValueError("O DataFrame de retornos (returns_wide) está vazio. Isso pode ocorrer se houver apenas um dia de dados nos preços ou se os preços forem constantes.")

        # Separar os dados do benchmark e dos ativos do portfólio
        self.benchmark_returns = returns_wide[self.benchmark_ticker]
        self.returns_data = returns_wide[self.tickers_list]
        self.prices_data = prices_wide[self.tickers_list]

        # Imprimir um resumo dos dados carregados
        print(f"Dados carregados com sucesso. Período analisado: {self.prices_data.index.min().strftime('%Y-%m-%d')} até {self.prices_data.index.max().strftime('%Y-%m-%d')}")
        print(f"Total de dias de negociação no período: {len(self.prices_data)}")
        print(f"Tickers incluídos no portfólio: {', '.join(self.tickers_list)}")
        print(f"Benchmark utilizado: {self.benchmark_ticker}")

    def calculate_portfolio_performance(self, weights):
        """
        Calcula o retorno esperado, a volatilidade (risco) e o Índice de Sharpe de um portfólio
        dados os pesos de alocação para cada ativo.

        Args:
            weights (np.array): Um array numpy contendo os pesos de cada ativo no portfólio.
                                A soma dos pesos deve ser 1 (ou muito próxima de 1).

        Returns:
            tuple: Uma tupla contendo:
                - portfolio_return (float): O retorno anualizado esperado do portfólio.
                - portfolio_volatility (float): A volatilidade anualizada do portfólio.
                - sharpe_ratio (float): O Índice de Sharpe anualizado do portfólio.
        """
        weights = np.array(weights) # Garante que os pesos sejam um array numpy

        # Retorno esperado do portfólio:
        # É a soma ponderada dos retornos médios históricos de cada ativo.
        # Multiplicamos por 252 para anualizar (considerando 252 dias de negociação no ano).
        portfolio_return = np.sum(self.returns_data.mean() * weights) * 252

        # Risco (volatilidade) do portfólio:
        # Calculado usando a matriz de covariância dos retornos dos ativos.
        # Volatilidade = sqrt(pesos_transpostos * matriz_covariancia * pesos)
        # A matriz de covariância também é anualizada multiplicando por 252.
        cov_matrix_annualized = self.returns_data.cov() * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix_annualized, weights)))

        # Índice de Sharpe:
        # Mede o retorno do portfólio em excesso à taxa livre de risco, por unidade de risco (volatilidade).
        # Sharpe Ratio = (Retorno do Portfólio - Taxa Livre de Risco) / Volatilidade do Portfólio
        # Se a volatilidade for zero (raro, mas possível com dados constantes ou um único ativo sem variação),
        # o Sharpe Ratio é definido como 0.0 para evitar divisão por zero.
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility != 0 else 0.0
        
        return portfolio_return, portfolio_volatility, sharpe_ratio

    
    def prepare_ml_features(self, window_size=30,target_window=10):
        """
        Prepara as features (variáveis de entrada) e targets (variáveis de saída)
        para os modelos de Machine Learning.

        Features Criadas (para cada ativo):
        -   Retornos acumulados em janelas de 5, 10, 20 e `window_size` dias.
        -   Volatilidade (desvio padrão dos retornos) em janelas de 10, 20 e `window_size` dias.
        -   Momentum (retorno acumulado nos últimos 90 dias).
        Features do Benchmark:
        -   Retorno acumulado do benchmark em 30 dias.
        -   Volatilidade do benchmark em 30 dias.

        Target (para cada ativo):
        -   Retorno futuro acumulado nos próximos 30 dias (shift(-30)).

        Args:
            window_size (int, optional): Tamanho da janela principal para cálculo de retornos e volatilidade.
                                       Padrão é 30 dias.

        Returns:
            tuple: Uma tupla contendo:
                - features_df (pd.DataFrame): DataFrame com as features para o ML.
                - targets_dict (dict): Dicionário onde as chaves são os tickers e os valores são Series
                                       com os retornos futuros (targets) para cada ativo.
                                       Retorna DataFrames vazios ou dicionários vazios se não houver dados suficientes.
        """
        print("\n--- Preparando Features para Modelos de Machine Learning ---")
        
        # Verifica se há dados de retorno suficientes para criar features
        if self.returns_data is None or self.returns_data.empty:
            print("Dados de retorno não disponíveis. Não é possível preparar features para ML.")
            return pd.DataFrame(), {}
        if len(self.returns_data) < 90 + window_size + 30: # Estimativa mínima para janelas e target futuro
            print(f"Dados de retorno insuficientes ({len(self.returns_data)} dias) para criar features e targets com as janelas especificadas. São necessários mais dias de histórico.")
            return pd.DataFrame(), {}

        features_df = pd.DataFrame(index=self.returns_data.index)

        # Criação de features para cada ativo no portfólio
        for ticker in self.tickers_list:
            # Retornos acumulados (soma dos retornos) em diferentes janelas
            for window in [5, 10, 20, window_size]:
                features_df[f'{ticker}_return_{window}d'] = self.returns_data[ticker].rolling(window=window).sum()
            # Volatilidade (desvio padrão dos retornos) em diferentes janelas
            for window in [10, 20, window_size]:
                features_df[f'{ticker}_vol_{window}d'] = self.returns_data[ticker].rolling(window=window).std()
            # Momentum (retorno acumulado em uma janela mais longa, ex: 90 dias)
            features_df[f'{ticker}_momentum_90d'] = self.returns_data[ticker].rolling(window=90).sum()

        # Adicionar features do benchmark
        if self.benchmark_returns is not None and not self.benchmark_returns.empty:
            features_df['market_return_30d'] = self.benchmark_returns.rolling(window=30).sum()
            features_df['market_vol_30d'] = self.benchmark_returns.rolling(window=30).std()
        else:
            print("Aviso: Dados de retorno do benchmark não disponíveis. Features de mercado não serão criadas.")

        # Remover linhas com valores NaN (gerados pelo rolling window no início da série)
        features_df = features_df.dropna()

        if features_df.empty:
            print("DataFrame de features ficou vazio após remover NaNs. Verifique o tamanho do histórico e as janelas.")
            return pd.DataFrame(), {}

        # Criação dos targets: retorno futuro de 30 dias para cada ativo
        targets_dict = {}
        for ticker in self.tickers_list:
            # O target é o retorno acumulado nos próximos 30 dias.
            # Usamos .shift(-30) para trazer os retornos futuros para a linha atual.
            future_returns = self.returns_data[ticker].rolling(window=target_window).sum().shift(-target_window)
            targets_dict[ticker] = future_returns.loc[features_df.index] # Alinha os targets com o índice das features
        
        # Concatenar targets para remover NaNs de forma consistente
        # Se algum target for NaN para uma data, essa data é removida das features e de todos os targets.
        target_df_combined = pd.concat(targets_dict, axis=1)
        valid_indices = ~target_df_combined.isna().any(axis=1)
        
        features_df = features_df.loc[valid_indices]
        targets_dict = {ticker: target_series.loc[valid_indices] for ticker, target_series in targets_dict.items()}

        if features_df.empty:
            print("DataFrame de features ficou vazio após alinhar com targets válidos. Pode não haver sobreposição suficiente entre features e targets futuros.")
            return pd.DataFrame(), {}
            
        print(f"Features e targets preparados. Shape das features: {features_df.shape}. Número de targets: {len(targets_dict)}.")
        return features_df, targets_dict

    
    def read_joblib(self):
        models_dict = {}
        
        for filename in os.listdir('api/model'):
            if filename.endswith('.joblib'):
                model_name = os.path.splitext(filename)[0].replace("ml_model_", "")
                model_path = os.path.join('api/model', filename)
                models_dict[model_name] = joblib.load(model_path)

        return models_dict
       

    def optimize_ml_portfolio(self, models_dict, features_df):
        """
        Otimiza o portfólio com base nas previsões de retorno dos modelos de Machine Learning.
        Os pesos são alocados proporcionalmente aos retornos positivos previstos.
        Se todas as previsões forem negativas ou zero, aloca os pesos igualmente.

        Args:
            models_dict (dict): Dicionário com os modelos de ML treinados para cada ativo.
            features_df (pd.DataFrame): DataFrame com as features (usará a última linha para fazer previsões).

        Returns:
            dict or None: Um dicionário com os resultados da otimização baseada em ML (método, pesos,
                          retorno, volatilidade, sharpe_ratio, previsões) se bem-sucedida.
                          Retorna None se não for possível otimizar (ex: sem modelos ou features).
        """
        print("\n--- Otimizando Portfólio com Base em Previsões de Machine Learning ---")
        if not models_dict or features_df.empty:
            print("Nenhum modelo de ML treinado ou features não disponíveis. Não é possível otimizar com ML.")
            return None
        
        # Usar as features mais recentes para fazer as previsões de retorno para o próximo período.
        latest_features = features_df.iloc[-1:].copy()
        
        predicted_returns = {}
        for ticker in self.tickers_list:
            if ticker in models_dict:
                predicted_returns[ticker] = models_dict[ticker].predict(latest_features)[0]
            else:
                # Se algum modelo não foi treinado (ex: por falta de dados), assume previsão de retorno zero.
                predicted_returns[ticker] = 0
        print(f"  Retornos previstos para o próximo período: {predicted_returns}")

        # Alocação de pesos baseada nas previsões:
        # Considera apenas os retornos previstos positivos.
        positive_predicted_returns = {ticker: ret for ticker, ret in predicted_returns.items() if ret > 0}
        total_positive_sum = sum(positive_predicted_returns.values())
        num_assets = len(self.tickers_list)

        weights_dict = {}
        if total_positive_sum > 0:
            # Aloca pesos proporcionalmente aos retornos previstos positivos.
            for ticker in self.tickers_list:
                weights_dict[ticker] = positive_predicted_returns.get(ticker, 0) / total_positive_sum
        else:
            # Se todos os retornos previstos são negativos ou zero, aloca igualmente (ou poderia ser caixa).
            print("  Todos os retornos previstos são negativos ou zero. Alocando pesos igualmente.")
            for ticker in self.tickers_list:
                weights_dict[ticker] = 1.0 / num_assets if num_assets > 0 else 0
        
        # Converter o dicionário de pesos para um array numpy para cálculo de performance.
        weight_array = np.array([weights_dict.get(ticker, 0) for ticker in self.tickers_list])
        
        # Normalização final para garantir que a soma dos pesos seja 1 (devido a arredondamentos ou lógica).
        if not np.isclose(np.sum(weight_array), 1.0) and np.sum(weight_array) > 0:
            weight_array = weight_array / np.sum(weight_array)
            weights_dict = {ticker: weight_array[i] for i, ticker in enumerate(self.tickers_list)}

        # Calcular as métricas de desempenho do portfólio com os pesos definidos pelo ML.
        # Nota: O retorno e volatilidade aqui são baseados em dados históricos, usando os pesos do ML.
        # O 'retorno' previsto pelo ML é usado para definir os pesos, não para o cálculo direto de performance histórica.
        portfolio_return, portfolio_volatility, sharpe_ratio = self.calculate_portfolio_performance(weight_array)
        
        print("Otimização com ML concluída!")
        print(f"  Retorno Esperado Anual do Portfólio (histórico com pesos ML): {portfolio_return:.4f}")
        print(f"  Volatilidade Anual do Portfólio (histórico com pesos ML): {portfolio_volatility:.4f}")
        print(f"  Índice de Sharpe Anualizado (histórico com pesos ML): {sharpe_ratio:.4f}")
        print("  Pesos Ótimos (baseados em ML):")
        for ticker, weight in weights_dict.items():
            print(f"    {ticker}: {weight:.4f}")

        return {
            'method': 'Machine Learning',
            'weights': weights_dict,
            'returns': portfolio_return, # Performance histórica com os pesos do ML
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe_ratio,
            'predicted_returns_for_allocation': predicted_returns # Retornos que o ML previu para definir os pesos
        }

    def backtest_portfolio(self, weights_input, start_date_str=None, end_date_str=None):
        """
        Realiza o backtesting de um portfólio com uma dada alocação de pesos.
        Calcula o desempenho histórico do portfólio e o compara com o benchmark.

        Args:
            weights_input (dict or np.array): Pesos do portfólio. Pode ser um dicionário {ticker: peso}
                                              ou um array numpy de pesos na ordem de `self.tickers_list`.
            start_date_str (str, optional): Data de início para o período de backtest (formato 'YYYY-MM-DD').
                                            Se None, usa o início dos dados de retorno disponíveis.
            end_date_str (str, optional): Data de fim para o período de backtest (formato 'YYYY-MM-DD').
                                          Se None, usa o fim dos dados de retorno disponíveis.

        Returns:
            dict or None: Um dicionário contendo as métricas e séries temporais do backtest
                          (retornos acumulados, retorno total, anualizado, volatilidade, sharpe, drawdown).
                          Retorna None se o backtest não puder ser realizado.
        """
        print(f"\n--- Realizando Backtesting do Portfólio ---")
        
        if self.returns_data is None or self.returns_data.empty:
            print("Dados de retorno não disponíveis. Backtesting não pode ser realizado.")
            return None

        # Copiar dados para evitar modificar os originais da classe
        returns_data_filtered = self.returns_data.copy()
        benchmark_returns_filtered = self.benchmark_returns.copy()

        # Filtrar os dados de retorno pelo período de backtest especificado
        if start_date_str:
            start_date = pd.to_datetime(start_date_str)
            returns_data_filtered = returns_data_filtered.loc[returns_data_filtered.index >= start_date]
            benchmark_returns_filtered = benchmark_returns_filtered.loc[benchmark_returns_filtered.index >= start_date]
        if end_date_str:
            end_date = pd.to_datetime(end_date_str)
            returns_data_filtered = returns_data_filtered.loc[returns_data_filtered.index <= end_date]
            benchmark_returns_filtered = benchmark_returns_filtered.loc[benchmark_returns_filtered.index <= end_date]

        if returns_data_filtered.empty:
            print("Não há dados de retorno para o período de backtest especificado.")
            return None

        # Converter/validar os pesos para formato de array numpy
        if isinstance(weights_input, dict):
            # Garante que os pesos estejam na mesma ordem que self.tickers_list
            weight_array = np.array([weights_input.get(ticker, 0) for ticker in self.tickers_list])
            # Normaliza se a soma não for 1 (ex: se algum ticker do dict não estava em self.tickers_list)
            if not np.isclose(np.sum(weight_array), 1.0) and np.sum(weight_array) > 0:
                print("Aviso: Soma dos pesos do dicionário não é 1. Normalizando para o backtest.")
                weight_array = weight_array / np.sum(weight_array)
        elif isinstance(weights_input, (list, np.ndarray)):
            weight_array = np.array(weights_input)
        else:
            print("Formato de pesos inválido para backtesting. Deve ser dict ou array/list.")
            return None
        
        if len(weight_array) != len(self.tickers_list):
            print(f"Erro: Número de pesos ({len(weight_array)}) não corresponde ao número de tickers ({len(self.tickers_list)}) no portfólio.")
            return None

        # Calcular os retornos diários do portfólio no período de backtest
        # Isso é feito multiplicando os retornos diários de cada ativo pelo seu peso no portfólio e somando.
        portfolio_daily_returns = returns_data_filtered.dot(weight_array)
        
        # Calcular os retornos acumulados do portfólio
        # (1 + r1) * (1 + r2) * ... * (1 + rn)
        cumulative_portfolio_returns = (1 + portfolio_daily_returns).cumprod()

        # Alinhar e calcular os retornos acumulados do benchmark para o mesmo período
        benchmark_returns_aligned = benchmark_returns_filtered.reindex(portfolio_daily_returns.index).fillna(0)
        cumulative_benchmark_returns = (1 + benchmark_returns_aligned).cumprod()

        # Calcular métricas de desempenho do backtest
        total_return_portfolio = cumulative_portfolio_returns.iloc[-1] - 1
        num_days_backtest = len(cumulative_portfolio_returns)
        
        # Retorno anualizado: ( (1 + Retorno Total) ^ (252 / Número de Dias) ) - 1
        annual_return_portfolio = (1 + total_return_portfolio) ** (252 / num_days_backtest) - 1 if num_days_backtest > 0 else 0.0
        
        # Volatilidade anualizada: Desvio padrão dos retornos diários * sqrt(252)
        annual_volatility_portfolio = portfolio_daily_returns.std() * np.sqrt(252)
        
        # Índice de Sharpe anualizado
        sharpe_ratio_portfolio = (annual_return_portfolio - self.risk_free_rate) / annual_volatility_portfolio if annual_volatility_portfolio != 0 else 0.0
        
        # Máximo Drawdown: Maior queda percentual do pico ao vale durante o período.
        rolling_max_portfolio = cumulative_portfolio_returns.cummax()
        drawdown_portfolio = (cumulative_portfolio_returns - rolling_max_portfolio) / rolling_max_portfolio
        max_drawdown_portfolio = drawdown_portfolio.min() if not drawdown_portfolio.empty else 0.0

        # Imprimir resultados do backtest
        actual_start_date = returns_data_filtered.index.min().strftime('%Y-%m-%d')
        actual_end_date = returns_data_filtered.index.max().strftime('%Y-%m-%d')
        print(f"  Período do Backtest: {actual_start_date} até {actual_end_date} ({num_days_backtest} dias)")
        print(f"  Retorno Total do Portfólio: {total_return_portfolio:.4%}")
        print(f"  Retorno Anualizado do Portfólio: {annual_return_portfolio:.4%}")
        print(f"  Volatilidade Anualizada do Portfólio: {annual_volatility_portfolio:.4%}")
        print(f"  Índice de Sharpe do Portfólio: {sharpe_ratio_portfolio:.4f}")
        print(f"  Máximo Drawdown do Portfólio: {max_drawdown_portfolio:.4%}")

        backtest_results = {
            'cumulative_returns_portfolio': cumulative_portfolio_returns,
            'cumulative_returns_benchmark': cumulative_benchmark_returns,
            'total_return': total_return_portfolio,
            'annual_return': annual_return_portfolio,
            'volatility': annual_volatility_portfolio,
            'sharpe_ratio': sharpe_ratio_portfolio,
            'max_drawdown': max_drawdown_portfolio,
            'start_date': actual_start_date,
            'end_date': actual_end_date,
            'num_days': num_days_backtest
        }
        
        # Plotar gráfico de retornos acumulados
        plt.figure(figsize=(12, 6))
        cumulative_portfolio_returns.plot(label='Portfólio Otimizado', legend=True)
        cumulative_benchmark_returns.plot(label=f'Benchmark ({self.benchmark_ticker})', legend=True, linestyle='--')
        plt.title(f'Desempenho Acumulado do Portfólio vs. Benchmark ({actual_start_date} a {actual_end_date})')
        plt.ylabel('Retorno Acumulado')
        plt.xlabel('Data')
        plt.grid(True)
        plt.show()

        return backtest_results

    def save_results(self, optimization_results_dict, backtest_results_dict, filename='portfolio_results.json'):
        """
        Salva os resultados da otimização e do backtest em um arquivo JSON.

        Args:
            optimization_results_dict (dict): Dicionário com os resultados da fase de otimização.
            backtest_results_dict (dict): Dicionário com os resultados da fase de backtest.
            filename (str, optional): Nome do arquivo JSON para salvar os resultados. 
                                      Padrão é 'portfolio_results.json'.

        Returns:
            dict or None: O dicionário combinado que foi salvo em JSON, ou None se os resultados forem None.
        """
        if optimization_results_dict is None or backtest_results_dict is None:
            print(f"Não há resultados de otimização ou backtest para salvar em {filename}.")
            return None
        
        # Preparar os dados para serem serializáveis em JSON.
        # Series do Pandas precisam ser convertidas para dicionários.
        output_data = {
            'optimization_results': optimization_results_dict,
            'backtest_metrics': { # Apenas as métricas escalares do backtest
                key: value for key, value in backtest_results_dict.items() 
                if not isinstance(value, pd.Series)
            },
            # Salvar as séries temporais de retornos acumulados como dicionários (data: valor)
            # As chaves do dicionário (datas) são convertidas para string no formato ISO.
            'cumulative_returns_portfolio_ts': {
                idx.isoformat(): val for idx, val in backtest_results_dict.get('cumulative_returns_portfolio', pd.Series()).items()
            },
            'cumulative_returns_benchmark_ts': {
                idx.isoformat(): val for idx, val in backtest_results_dict.get('cumulative_returns_benchmark', pd.Series()).items()
            }
        }
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
            print(f"Resultados da otimização e backtest salvos com sucesso em: {filename}")
            return output_data
        except Exception as e:
            print(f"Erro ao salvar os resultados em JSON: {e}")
            return None







def main_run():
    """
    Função principal de exemplo para demonstrar o uso da classe PortfolioOptimizer.
    Esta função configura os parâmetros, cria/carrega os dados, executa as otimizações
    e os backtests, e salva os resultados.
    """
    print("\n" + "="*80)
    print("INICIANDO EXEMPLO DE EXECUÇÃO DO OTIMIZADOR DE PORTFÓLIO")
    print("="*80 + "\n")

    # --- 1. Configuração dos Parâmetros do Portfólio ---
    # Lista de tickers dos ativos que você deseja incluir no seu portfólio.
    # Exemplo: Ações de tecnologia e o S&P 500 como benchmark.
    USER_TICKERS = ['AAPL', 'GOOG', 'AMZN', 'NFLX', 'MSFT', 'IBM']
    BENCHMARK_TICKER = '^GSPC' # Ticker do S&P 500 no Yahoo Finance
    ALL_TICKERS_FOR_DATA_SIMULATION = USER_TICKERS + [BENCHMARK_TICKER]

    # Período para os dados históricos
    START_DATE_DATA = '2007-01-01'
    END_DATE_DATA = '2025-04-30'

   
    #Para usar a base localmente


    df_total = pd.read_csv('datas/dados_base.csv')

    #Ajustar os tipos de colunas
    df_total['Date'] = pd.to_datetime(df_total['Date'])
    df_total['Close'] = df_total['Close'].astype(float)
    df_total['High'] = df_total['High'].astype(float)
    df_total['Low'] = df_total['Low'].astype(float)
    df_total['Open'] = df_total['Open'].astype(float)
    df_total['Volume'] = df_total['Volume'].astype(int)
    df_total['ticker'] = df_total['ticker'].astype(str)


    # --- 3. Inicialização do Otimizador ---
    try:
        # Cria uma instância da classe PortfolioOptimizer, passando os dados e parâmetros.
        optimizer = PortfolioOptimizer(df_total_data=df_total,
                                       tickers_list=USER_TICKERS,
                                       benchmark_ticker=BENCHMARK_TICKER,
                                       risk_free_rate=RISK_FREE_RATE)
    except ValueError as e:
        print(f"Erro ao inicializar o PortfolioOptimizer: {e}")
        print("Verifique os dados de entrada e os tickers fornecidos.")
        return

    # --- 4. Otimização de Markowitz ---
    markowitz_results = optimizer.optimize_markowitz()

    # --- 5. Preparação e Treinamento dos Modelos de ML (Opcional) ---
    # Inicializa variáveis para o caso de a etapa de ML ser pulada.
    features_df = pd.DataFrame()
    targets_dict = {}
    ml_models_dict = {}
    ml_optimization_results = None

    # Verifica se há dados suficientes para as janelas de features e target do ML.
    # Este valor (ex: 150) é uma heurística; pode precisar de ajuste.
    MIN_DAYS_FOR_ML = 90 + 30 + 30 + 60 # Aprox: maior janela feature + janela target + período de teste
    if len(optimizer.returns_data) < MIN_DAYS_FOR_ML:
         print(f"\nDados de retorno insuficientes ({len(optimizer.returns_data)} dias) para a etapa de Machine Learning completa. Pulando ML.")
         print(f"São necessários pelo menos {MIN_DAYS_FOR_ML} dias de histórico para as configurações atuais de features/target/split.")
    else:
        features_df, targets_dict = optimizer.prepare_ml_features()
        if not features_df.empty and targets_dict:
            ml_models_dict, _ml_performance = optimizer.train_ml_models(features_df, targets_dict)
            if ml_models_dict:
                # --- 6. Otimização com Base em ML ---
                ml_optimization_results = optimizer.optimize_ml_portfolio(ml_models_dict, features_df)
            else:
                print("Não foi possível treinar os modelos de ML. A otimização baseada em ML será pulada.")
        else:
            print("Não foi possível preparar features ou targets para ML. A otimização baseada em ML será pulada.")

    # --- 7. Backtesting das Estratégias ---
    # Definir a data de início do backtest.
    # Idealmente, o backtest deve começar após o período usado para treinar os modelos de ML
    # ou após um período inicial de estabilização dos dados para Markowitz.
    # Se features de ML foram criadas, usamos a primeira data delas como início do backtest.
    backtest_start_date = None
    if not features_df.empty:
        backtest_start_date = features_df.index[0].strftime('%Y-%m-%d')
        print(f"\nDefinindo data de início do backtest como {backtest_start_date} (início das features de ML).")
    elif not optimizer.returns_data.empty and len(optimizer.returns_data) > 252: # Pelo menos 1 ano de dados
        # Fallback: Se não houver features de ML, inicia o backtest após o primeiro ano de dados.
        backtest_start_date = optimizer.returns_data.index[252].strftime('%Y-%m-%d')
        print(f"\nAviso: Features de ML não disponíveis. Definindo data de início do backtest como {backtest_start_date} (após 1 ano de dados).")
    else:
        print("\nAviso: Não foi possível determinar uma data de início adequada para o backtest. O backtest pode usar todo o período de dados.")

    # Backtest para Markowitz
    if markowitz_results and markowitz_results.get('weights'):
        print(f"\nExecutando backtest para a estratégia de Markowitz...")
        markowitz_backtest_results = optimizer.backtest_portfolio(
            weights_input=markowitz_results['weights'], 
            start_date_str=backtest_start_date
        )
        if markowitz_backtest_results:
            optimizer.save_results(markowitz_results, markowitz_backtest_results, 'markowitz_full_script_results.json')
    else:
        print("\nNão há resultados da otimização de Markowitz para realizar backtest.")
    
    # Backtest para Machine Learning
    if ml_optimization_results and ml_optimization_results.get('weights'):
        print(f"\nExecutando backtest para a estratégia baseada em Machine Learning...")
        ml_backtest_results = optimizer.backtest_portfolio(
            weights_input=ml_optimization_results['weights'], 
            start_date_str=backtest_start_date
        )
        if ml_backtest_results:
            optimizer.save_results(ml_optimization_results, ml_backtest_results, 'ml_full_script_results.json')
    else:
        print("\nNão há resultados da otimização baseada em ML para realizar backtest.")

    print("\n" + "="*80)
    print("EXECUÇÃO DO EXEMPLO DO OTIMIZADOR DE PORTFÓLIO CONCLUÍDA")
    print("="*80 + "\n")


if __name__ == "__main__":
    main_run()

