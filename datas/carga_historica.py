import yfinance as yf
import pandas as pd
import boto3
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

def buscar_dados_historicos(tickers, start, end, intervalo='1d'):
    frames = []

    for ticker in tickers:
        print(f"Baixando dados de {ticker} de {start} até {end}...")
        df = yf.download(ticker, start=start, end=end, interval=intervalo)

        if df.empty:
            print(f"Nenhum dado retornado para {ticker}.")
            continue

        # Ajusta colunas, se necessário
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

def salvar_em_s3_particionado(df, bucket):
    if 'Date' not in df.columns:
        raise ValueError("Coluna 'Date' não encontrada no DataFrame.")

    df['Date'] = pd.to_datetime(df['Date'])
    s3 = boto3.client( 's3',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
    )

    for (ticker, ano), grupo in df.groupby(["ticker", df['Date'].dt.year]):
        path = f"dados_financeiros/ticker={ticker}/year={ano}/dados.parquet"

        parquet_buffer = BytesIO()
        grupo.to_parquet(parquet_buffer, index=False, engine='pyarrow', compression='snappy')
        parquet_buffer.seek(0)

        try:
            s3.put_object(Bucket=bucket, Key=path, Body=parquet_buffer.getvalue())
            print(f"Salvo: s3://{bucket}/{path}")
        except Exception as e:
            print(f"Erro ao salvar {path}: {e}")

def executar_pipeline_historica():
    tickers = ['AAPL','GOOG','AMZN', 'NFLX', 'MSFT', 'IBM','^GSPC'] 
    start = "2007-01-01"  
    end = datetime.now().strftime("%Y-%m-%d") 
    bucket = "fiap-tch3-mlet"  

    df = buscar_dados_historicos(tickers, start, end, intervalo="1d")
    if not df.empty:
        print("Estrutura final dos dados:")
        print(df[['Date', 'ticker', 'Open', 'Close']].head())

        salvar_em_s3_particionado(df, bucket)
    else:
        print("Nenhum dado para salvar.")

if __name__ == "__main__":
    executar_pipeline_historica()



