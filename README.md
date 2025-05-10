# Tech Challenge Fase 3 - Ingestão de dados + ML

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
📂 data/                                    # Script que faz a consulta dos dados do yfinance e alimenta o data lake na s3
📂 models/                                  # Scripts para modelos de previsão e resultados
📄 requirements.txt                         # Dependências do projeto
📄 EDA.py                                   # Script com análise exploratória completa
📄 Análise e Interpretação.md               # Análise e Interpretação dos resultados do modelo escolhido
📄 README.md                                # Documentação do projeto
```

---

## Como executar o projeto

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/nome-do-repo.git
cd nome-do-repo
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

Esta etapa foi feita no notebook EDA.ipynb, onde exploramos a base histórica, trouxemos novas variáveis e visualizações para emabasar a modelagem pretidiva

### 5. Modelagem

Em /models, temos 3 arquivos, dois foram modelos de ML para previsão de preço e previsão de retorno.
O modelo escolhido está no script models/ML_otimizacao_portfolio.py e é um modelo de otimização de portfólio.

Testamos dois modelos distintos: Markowitx e ML (Random Forest Regressor) e fizemos análises comparativas entre eles.

Para executar

```bash
python models/ML_otimizacao_portfolio.py
```


## 📊 Principais Entregas

- Análise exploratória completa com:
  - Retorno, volatilidade, drawdown, correlação e decomposição
  - Comparativo antes/depois de 2020
  - Cálculo e visualização do índice de Sharpe

- Dois modelos testados:
  - **Markowitz (Modern Portfolio Theory)**
  - **Random Forest Regressor (ML)**

---

