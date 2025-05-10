import yfinance as yf
import pandas as pd
import boto3
from io import BytesIO
from datetime import datetime
import pytz

# Parâmetros globais
TICKERS = ['AAPL','GOOG','AMZN', 'NFLX', 'MSFT', 'IBM','^GSPC']  
BUCKET_NAME = "fiap-tch3-mlet"
INTERVAL = "1m"  # Coleta por minuto
START_OFFSET_MINUTES = 2  # Evita pegar minuto atual incompleto

def lambda_handler(event, context):
    now = datetime.now(pytz.timezone("America/Sao_Paulo"))
    start = now.replace(second=0, microsecond=0) - pd.Timedelta(minutes=START_OFFSET_MINUTES)
    end = now

    frames = []
    for ticker in TICKERS:
        print(f"📥 Coletando {ticker} de {start} até {end}")
        df = yf.download(ticker, start=start, end=end, interval=INTERVAL)

        if df.empty:
            print(f"⚠️ Nenhum dado retornado para {ticker}")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.reset_index(inplace=True)
        df.rename(columns={"Datetime": "Date"}, inplace=True)
        df['ticker'] = ticker
        frames.append(df)

    if not frames:
        return {"statusCode": 204, "body": "Sem dados coletados"}

    df_final = pd.concat(frames, ignore_index=True)

    s3 = boto3.client("s3")

    # Agrupar por partições temporais: ano/mês/dia/hora/minuto
    for (ano, mes, dia, hora, minuto), grupo in df_final.groupby([
        df_final['Date'].dt.year,
        df_final['Date'].dt.month,
        df_final['Date'].dt.day,
        df_final['Date'].dt.hour,
        df_final['Date'].dt.minute
    ]):
        path = (
            f"dados_tickers/granularidade=1min/"
            f"year={ano}/month={mes:02d}/day={dia:02d}/hour={hora:02d}/minute={minuto:02d}/"
        )
        file_name = "dados.parquet"

        parquet_buffer = BytesIO()
        grupo.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)

        s3.put_object(Bucket=BUCKET_NAME, Key=path + file_name, Body=parquet_buffer.getvalue())
        print(f"✅ Salvo: s3://{BUCKET_NAME}/{path}{file_name}")

    return {"statusCode": 200, "body": f"{len(df_final)} registros salvos"}
