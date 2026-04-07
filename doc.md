# Documentação da Solução

## Objetivo
Criar uma API simples para prever se um cliente do banco tem chance de assinar um depósito a prazo.

## Arquitetura
- `scripts/train_model.py`: baixa o dataset, trata os dados, treina o modelo e salva o arquivo `.joblib`.
- `app/schemas.py`: valida os dados de entrada e saída da API.
- `app/model.py`: carrega o modelo e faz a predição.
- `app/main.py`: expõe os endpoints `/health`, `/info` e `/predict`.
- `tests/`: testes unitários e de integração.

## Escolhas Técnicas
- **FastAPI**: API simples e rápida.
- **Pydantic**: validação dos dados de entrada.
- **Scikit-Learn**: treino do modelo com pipeline.
- **RandomForestClassifier**: algoritmo fácil de usar e com bom resultado inicial.
- **Docker**: facilita execução em qualquer máquina.

## Fluxo de Dados
1. O script de treino baixa o dataset Bank Marketing.
2. Os dados passam por pré-processamento.
3. O modelo é treinado e salvo em `models/bank_marketing_model.joblib`.
4. A API carrega o modelo ao iniciar.
5. O usuário envia um JSON no endpoint `/predict`.
6. A API devolve a predição e a probabilidade.

## Validação
O schema bloqueia campos extras e valida:
- idade
- saldo
- dia do contato
- duração
- campanha
- categorias como `job`, `marital`, `education`, `month`

## Estratégia de Testes
- Teste unitário do schema válido.
- Teste de schema inválido.
- Teste de campo extra.
- Teste do endpoint `/health`.
- Teste do endpoint `/predict`.
- Teste negativo com payload inválido.
- Teste do caso em que o modelo não existe.

## Como Executar
```bash
pip install -r requirements.txt
python scripts/train_model.py
pytest tests/ -v
uvicorn app.main:app --reload
```

## Docker
```bash
docker build -t bank-marketing-api .
docker run -p 8000:8000 bank-marketing-api
```
