import yfinance as yf
import pandas as pd
from datetime import datetime
import os

def buscar_dados_historicos(tickers, start, end, intervalo='1d'):
    frames = []

    for ticker in tickers:
        print(f"Baixando dados de {ticker} de {start} até {end}...")
        df = yf.download(ticker, start=start, end=end, interval=intervalo)

        if df.empty:
            print(f"Nenhum dado retornado para {ticker}.")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.copy()
        df.reset_index(inplace=True)
        df['ticker'] = ticker

        frames.append(df)

    if frames:
        df_final = pd.concat(frames, ignore_index=True)
        return df_final
    else:
        print("Nenhum dado foi baixado.")
        return pd.DataFrame()

def salvar_localmente(df, caminho='datas/dados_base.csv'):
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    df.to_csv(caminho, index=False)
    print(f"Base salva localmente em: {caminho}")

def executar_pipeline_local():
    tickers = ['AAPL','GOOG','AMZN', 'NFLX', 'MSFT', 'IBM','^GSPC']
    start = "2007-01-01"
    end = datetime.now().strftime("%Y-%m-%d")

    df = buscar_dados_historicos(tickers, start, end, intervalo="1d")
    if not df.empty:
        print("Estrutura final dos dados:")
        print(df[['Date', 'ticker', 'Open', 'Close']].head())

        salvar_localmente(df)
    else:
        print("Nenhum dado para salvar.")

if __name__ == "__main__":
    executar_pipeline_local()
