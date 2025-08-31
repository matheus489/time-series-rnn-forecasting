# ⚡ Instalação Rápida - Sistema de Previsão de Séries Temporais

## 🎯 Instalação em 3 Passos

### Passo 1: Pré-requisitos
```bash
# Verifique se tem Python 3.8+
python --version

# Se não tiver Python, baixe de: https://python.org
```

### Passo 2: Instalar Dependências
```bash
# Instale todas as bibliotecas necessárias
pip install -r requirements.txt
```

### Passo 3: Testar Instalação
```bash
# Execute o sistema
python run_demo.py
```

## 🚀 Primeira Execução

### Opção A: Demonstração Automática
```bash
python quick_start.py
```
- ✅ Funciona imediatamente
- 🔄 Usa dados sintéticos
- 📊 Mostra resultados completos

### Opção B: Menu Interativo
```bash
python run_demo.py
```
Escolha:
- **Opção 6**: "Executar tudo" (recomendado)
- **Opção 4**: "Executar demonstração"
- **Opção 5**: "Iniciar aplicação Streamlit"

## 🔑 Configuração de APIs (Opcional)

### Para Dados Reais

#### 1. Configure API Keys
```bash
python setup_api_keys.py
```

#### 2. Registre-se nas APIs (gratuito)
- **Alpha Vantage**: https://www.alphavantage.co/
- **Polygon.io**: https://polygon.io/
- **Finnhub**: https://finnhub.io/

#### 3. Teste as APIs
```bash
python test_data_sources.py
```

## 📊 O que você verá

### Resultados Esperados
```
🚀 Início Rápido - Sistema de Previsão de Séries Temporais
============================================================

1. 📊 Carregando dados de ações...
✅ Dados carregados para AAPL: 249 registros

2. 🔧 Preparando dados...
✅ Dados preparados: 175 amostras de treino

3. 🎯 Treinando modelo LSTM...
✅ Modelo treinado com sucesso!

4. 📈 Avaliando modelo...
R² Score: 0.5007
RMSE: 2.8527
MAPE: 1.30%

5. 📊 Resultados salvos em plots/
```

### Gráficos Gerados
- 📈 Preços históricos vs predições
- 📊 Métricas de performance
- 🔮 Previsões futuras
- 📉 Histórico de treinamento

## 🔧 Solução de Problemas

### Erro: "ModuleNotFoundError"
```bash
# Reinstale dependências
pip install -r requirements.txt
```

### Erro: "API key inválida"
```bash
# Use dados sintéticos (funciona automaticamente)
python quick_start.py
```

### Erro: "Dados insuficientes"
```bash
# Configure APIs ou use dados sintéticos
python setup_api_keys.py
```

### Erro: "Timeout"
```bash
# Verifique conexão com internet
# O sistema usa fallback automático
```

## 📁 Estrutura Após Instalação

```
📁 Projeto/
├── 📁 data/                  # Dados baixados
│   ├── 📄 AAPL_raw_data.csv
│   └── 📄 AAPL_synthetic_data.csv
├── 📁 plots/                 # Gráficos gerados
│   ├── 📄 AAPL_lstm_predictions.png
│   └── 📄 training_history.png
├── 📁 models/                # Modelos salvos
└── 📄 api_config.json        # Configuração de APIs
```

## 🎮 Próximos Passos

### Para Iniciantes
1. **Execute**: `python quick_start.py`
2. **Explore**: `python run_demo.py`
3. **Configure**: `python setup_api_keys.py`

### Para Uso Avançado
1. **Interface Web**: `python run_demo.py` → Opção 5
2. **Jupyter Notebook**: `jupyter notebook notebooks/`
3. **Customização**: Edite arquivos em `src/`

### Para Desenvolvimento
1. **Modifique modelos**: `src/model.py`
2. **Adicione APIs**: `src/data_loader.py`
3. **Customize visualizações**: `src/utils.py`

## 📞 Suporte Rápido

### Comandos de Diagnóstico
```bash
# Testar sistema
python test_data_sources.py

# Verificar instalação
python -c "import torch, pandas, streamlit; print('✅ Tudo OK!')"

# Limpar cache (se necessário)
rm -rf data/* plots/* models/*
```

### Logs Importantes
- **Dados**: Verifique pasta `data/`
- **Gráficos**: Verifique pasta `plots/`
- **Erros**: Veja mensagens no terminal

## 🎉 Sucesso!

Se você viu:
- ✅ "Dados carregados"
- ✅ "Modelo treinado"
- ✅ "R² Score: X.XXXX"
- 📊 Gráficos na pasta `plots/`

**Parabéns! O sistema está funcionando perfeitamente!**

---

**💡 Dica**: Comece sempre com `python quick_start.py` para uma demonstração completa.
