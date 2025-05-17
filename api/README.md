# API Flask ML

Uma API baseada em Flask para predições de machine learning.

## Configuração

1. Criar um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # No Windows use: venv\Scripts\activate
```

2. Instalar dependências:
```bash
pip install -r requirements.txt
```

3. Criar arquivo `.env` (opcional):
```bash
PORT=5000
```

## Executando a API

Modo desenvolvimento:
```bash
python ml_api.py
```

Modo produção com Gunicorn:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 ml_api:app
```

## Endpoints da API

- `GET /health`: Endpoint de verificação de saúde
- `POST /api/v1/predict`: Endpoint de predição

### Exemplo de Requisição (predict)

```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [5.1, 3.5, 1.4, 0.2]}'
```

Exemplo de resposta:
```json
{
    "predicao": "setosa",
    "predicao_numerica": 0,
    "caracteristicas_entrada": [5.1, 3.5, 1.4, 0.2]
}
```

Os valores de entrada representam:
1. comprimento_sepala (cm)
2. largura_sepala (cm)
3. comprimento_petala (cm)
4. largura_petala (cm)

## Logs

Os logs são armazenados no diretório `logs` com rotação automática.