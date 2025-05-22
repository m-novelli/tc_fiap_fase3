# Tech Challenge Fase 3 - Ingestão de dados + ML + API

Projeto de pós-graduação em Machine Learning Engineering com foco em ingestão de dados, desenvolvimento de um modelo de ML e uso produtivo dos resultados

---

## Etapas do Projeto

- Uso dos dados do yfinance de tickers de tecnologia (2007-2025)
- Ingestão dos dados da API do yfinance em um data lake na s3
- Análise exploratória dos dados
- Desenvolvimento de Modelos de ML 
- API para devolver o resultado do modelo


---

## 📁 Estrutura do Projeto

```
📂 api/                                    # Aplicação que executa o modelo de ML através da API
📂 data/                                    # Script que faz a consulta dos dados do yfinance e alimenta o data lake na s3
📂 models/                                  # Analise Exploratoria e Scripts para modelos de previsão e resultados
📄 requirements.txt                         # Dependências do projeto
📄 Análise e Interpretação.md               # Análise e Interpretação dos resultados do modelo escolhido
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

```bash
pip install -r requirements.txt
```

### 3. Dados salvos localmente

Para maior facilidade em reproduzir as análise e Modelos, salvamos a base localmente em
datas/dados_base.csv


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

## Próximos Passos

Implementar um backtest com rebalanceamento periódico, onde os pesos são recalculados ao longo do tempo, com base apenas nas informações disponíveis até cada momento.

Treinar modelos para mais tickers, de forma que o usuário tenha liberdade de montar uma carteira mais diversificada e não somente baseada em tickers de tecnologia.

---

