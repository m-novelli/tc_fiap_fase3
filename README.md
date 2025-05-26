# Tech Challenge Fase 3 - Ingestão de dados + ML + API

Projeto de pós-graduação em Machine Learning Engineering com foco em ingestão de dados, desenvolvimento de um modelo de ML e uso produtivo dos resultados

---

## Etapas do Projeto

- Uso dos dados do yfinance de tickers de tecnologia (2007-2025)
- Ingestão dos dados da API do yfinance em um data lake na s3
- Análise exploratória dos dados
- Desenvolvimento de Modelos de ML
- Otimização de Portfólio
- API para devolver o resultado do modelo


---

## 📁 Estrutura do Projeto

```
📂 data/                                    # Script que faz a consulta dos dados do yfinance e alimenta o data lake na s3
📂 models/                                  # Analise Exploratoria e Scripts para modelos de previsão e resultados
📄 requirements.txt                         # Dependências do projeto
📄 Análise e Interpretação.md               # Análise e Interpretação dos resultados do modelo escolhido
📄 api.py                                   # Aplicação que executa o modelo de ML através da API
📄 README.md                                # Documentação do projeto
```

---

## Como executar o projeto

### 1. Clone o repositório

```bash
git clone https://github.com/m-novelli/tc_fiap_fase3
cd tc_fiap_fase3
```

### 2. Instale as dependências

Recomendado usar um ambiente virtual para evitar conflitos de dependências.

```bash
python -m venv ml_api_venv
source ml_api_venv/bin/activate  # No Windows use: ml_api_venv\Scripts\activate
```

Instalar dependências:
```bash
pip install -r requirements.txt
```

### 3. Dados salvos localmente

Para o treinamento dos modelos, usamos script para baixar os dados do yfinance e salvar localmente em datas/dados_base.csv (caminho configurável). No momento da predição, o caminho dos dados utilizados é datas/dados_base_predict.csv.

### 3.1 Alternativa: Usar base do s3

Adicione as credenciais do AWS CLI presentes no AWS Details do AWS Academy Lab no arquivo .aws/credentials
Descomentar as partes que fazem leitura da base local e ajustar para ler direto do s3

```bash
python datas/carga_historica.py
```

### 4. Análise exploratória

Esta etapa foi feita no notebook 0.Analise Exploratoria.ipynb, onde exploramos a base histórica, trouxemos novas variáveis e visualizações para emabasar a modelagem pretidiva

### 5. Modelagem

Foram testados modelos como ARIMA, Random Forest e XGBoost para previsão de preço e retorno de ativos.
O modelo Random Forest foi selecionado para previsão de retornos e passou por otimização de hiperparâmetros via Grid Search com validação temporal (TimeSeriesSplit), visando maximizar o desempenho preditivo em séries temporais.

Modelo Final – Otimização de Portfólio com ML:
A estratégia escolhida foi a otimização de portfólio com base nas previsões semanais de retorno geradas pelo modelo Random Forest ajustado.
As previsões alimentam uma rotina de alocação ótima com foco em maximizar o retorno esperado ajustado ao risco.


## Principais Entregas

- Análise exploratória completa com:
- Treinamento e seleção de Modelo de ML
- Otimização de Portfólio com backtesting e cálculos de performance
- Dump do modelo selecionado em .joblib para consumo da API
- API que dá os pesos ótimos de cada ticker de tecnologia para maior retorno.

# API Flask ML

Uma API baseada em Flask para retornar os pesos ótimos de cada ticker para maior retorno.

1. Criar arquivo `.env` (opcional):
```bash
PORT=5000
```

## Executando a API

```bash
python api.py
```

## Endpoints da API

- `GET /health`: Endpoint de verificação de saúde
- `POST /api/v1/predict`: Endpoint de predição

### Exemplo de Requisição (predict)

```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '
  {
    "user_tickers": ["AAPL", "GOOG", "AMZN", "NFLX", "MSFT", "IBM"],
    "benchmark_ticker": "^GSPC",
    "start_date": "2020-01-01",
    "end_date": "2025-05-15"
}
'
```

Exemplo de resposta:
```json
{
    "prediction": {
        "method": "Machine Learning",
        "predicted_returns_for_allocation": {
            "AAPL": 0.021008336035190856,
            "AMZN": 0.016394787872220068,
            "GOOG": 0.01177844021958414,
            "IBM": 0.004847179685692798,
            "MSFT": 0.007040221264398282,
            "NFLX": 0.018708126613573267
        },
        "returns": 0.2584574456282873,
        "sharpe_ratio": 0.8361334199410123,
        "volatility": 0.28519066448164465,
        "weights": {
            "AAPL": 0.26333795316394804,
            "AMZN": 0.20550746492228456,
            "GOOG": 0.1476418852827547,
            "IBM": 0.06075904226351138,
            "MSFT": 0.08824865779385858,
            "NFLX": 0.23450499657364274
        }
    }
}
```

## Logs

Os logs são armazenados no diretório `logs` com rotação automática.

## Próximos Passos

Implementar um backtest com rebalanceamento periódico, onde os pesos são recalculados ao longo do tempo, com base apenas nas informações disponíveis até cada momento.

Treinar modelos para mais tickers, de forma que o usuário tenha liberdade de montar uma carteira mais diversificada e não somente baseada em tickers de tecnologia.

Tornar a lista de tickers dinâmica, para que o usuário possa escolher quais tickers deseja incluir na carteira.
---

