# 📈 Sistema de Previsão de Séries Temporais com RNN

Um sistema completo e robusto para previsão de preços de ações usando Redes Neurais Recorrentes (RNN), com suporte a múltiplas fontes de dados financeiros e interface interativa.

## 🎯 Visão Geral

Este projeto implementa um sistema avançado de previsão de séries temporais que utiliza diferentes arquiteturas de Redes Neurais Recorrentes (LSTM, GRU, RNN) para prever preços de ações. O sistema é projetado para ser robusto, flexível e fácil de usar, com suporte a dados reais de múltiplas APIs financeiras.

### ✨ Características Principais

- **🔗 Múltiplas Fontes de Dados**: Yahoo Finance, Alpha Vantage, Polygon.io, Finnhub, IEX Cloud, Quandl
- **🧠 Modelos Avançados**: LSTM, GRU, RNN com diferentes configurações
- **📊 Visualizações Interativas**: Gráficos dinâmicos com Plotly e Streamlit
- **🔄 Sistema Robusto**: Fallback automático para dados sintéticos
- **⚙️ Configuração Flexível**: Parâmetros ajustáveis para todos os modelos
- **📈 Métricas Completas**: MSE, RMSE, MAE, R², MAPE
- **🎮 Interface Amigável**: Menu interativo e aplicação web

## 🏗️ Arquitetura do Sistema

```
📁 Projeto/
├── 📁 src/                    # Código fonte principal
│   ├── 📄 data_loader.py      # Carregador de dados com múltiplas APIs
│   ├── 📄 model.py           # Definições dos modelos RNN
│   ├── 📄 trainer.py         # Treinamento e avaliação
│   ├── 📄 utils.py           # Utilitários e visualizações
│   └── 📄 streamlit_app.py   # Interface web interativa
├── 📁 notebooks/             # Jupyter notebooks
│   ├── 📄 analise_series_temporais.ipynb  # Análise completa
│   └── 📄 demo_analysis.py   # Demonstração em Python
├── 📁 data/                  # Dados baixados e processados
├── 📁 models/                # Modelos treinados salvos
├── 📁 plots/                 # Gráficos gerados
├── 📄 quick_start.py         # Início rápido do sistema
├── 📄 run_demo.py           # Menu interativo principal
├── 📄 test_data_sources.py  # Teste das fontes de dados
├── 📄 setup_api_keys.py     # Configuração de API keys
├── 📄 api_config.json       # Configuração das APIs
└── 📄 requirements.txt      # Dependências do projeto
```

## 🚀 Como Usar

### 1. Instalação e Configuração

#### Pré-requisitos
- Python 3.8+
- Conexão com internet (para baixar dados)

#### Instalação
```bash
# Clone o repositório
git clone <url-do-repositorio>
cd previsao-series-temporais-rnn

# Instale as dependências
pip install -r requirements.txt
```

#### Configuração de API Keys (Opcional)
Para obter dados reais, configure suas API keys:

```bash
# Configuração automática
python setup_api_keys.py

# Ou crie manualmente o arquivo api_config.json:
{
  "alpha_vantage": "SUA_API_KEY",
  "polygon": "SUA_API_KEY", 
  "finnhub": "SUA_API_KEY"
}
```

**APIs Gratuitas Disponíveis:**
- **Alpha Vantage**: 500 requisições/dia (Recomendado)
- **Polygon.io**: 5 requisições/minuto
- **Finnhub**: 60 requisições/minuto

### 2. Execução do Sistema

#### Opção 1: Menu Interativo (Recomendado)
```bash
python run_demo.py
```
Este comando abre um menu com todas as opções disponíveis:
- 📦 Instalar dependências
- 📊 Carregar dados
- 🎯 Treinar modelo
- 📓 Executar demonstração
- 🌐 Iniciar aplicação Streamlit
- 🚀 Executar tudo

#### Opção 2: Início Rápido
```bash
python quick_start.py
```
Executa uma demonstração completa automaticamente.

#### Opção 3: Interface Web (Streamlit)
```bash
python run_demo.py
# Escolha opção 5: Iniciar aplicação Streamlit
```
Abre uma interface web interativa no navegador.

#### Opção 4: Jupyter Notebook
```bash
jupyter notebook notebooks/analise_series_temporais.ipynb
```
Para análise detalhada e experimentação.

## 📊 Fontes de Dados

### 🔗 APIs Suportadas

O sistema tenta baixar dados na seguinte ordem:

1. **Yahoo Finance** (Sem API key necessária)
2. **Alpha Vantage** (500 req/dia gratuitas)
3. **Polygon.io** (5 req/min gratuitas)
4. **Finnhub** (60 req/min gratuitas)
5. **IEX Cloud** (50k msg/mês gratuitas)
6. **Quandl** (50 req/dia gratuitas)

### 🔄 Sistema de Fallback

Se todas as APIs falharem, o sistema automaticamente:
- Gera dados sintéticos realistas
- Continua funcionando normalmente
- Permite demonstração e aprendizado

### 📈 Dados Disponíveis

Para cada ação, o sistema baixa:
- **Preços**: Open, High, Low, Close
- **Volume**: Volume de negociação
- **Períodos**: 1m, 3m, 6m, 1y, 2y, 5y
- **Ações**: AAPL, GOOGL, MSFT, TSLA, AMZN, etc.

## 🧠 Modelos de Machine Learning

### Arquiteturas Implementadas

#### 1. LSTM (Long Short-Term Memory)
- **Vantagens**: Excelente para séries temporais longas
- **Uso**: Padrão para previsão de ações
- **Configuração**: Hidden layers, dropout, sequence length

#### 2. GRU (Gated Recurrent Unit)
- **Vantagens**: Mais eficiente que LSTM
- **Uso**: Alternativa rápida ao LSTM
- **Configuração**: Similar ao LSTM, menos parâmetros

#### 3. RNN (Recurrent Neural Network)
- **Vantagens**: Simples e rápido
- **Uso**: Comparação e baseline
- **Configuração**: Básica, poucos parâmetros

### Parâmetros Configuráveis

```python
# Exemplo de configuração
model_params = {
    'model_type': 'lstm',      # 'lstm', 'gru', 'rnn'
    'hidden_size': 50,         # 20-100
    'num_layers': 2,           # 1-4
    'dropout': 0.2,            # 0.0-0.5
    'sequence_length': 60      # 30-120
}

training_params = {
    'epochs': 50,              # 10-200
    'batch_size': 32,          # 16-64
    'learning_rate': 0.001,    # 0.0001-0.01
    'patience': 15             # Early stopping
}
```

## 📈 Métricas de Avaliação

### Métricas Implementadas

1. **MSE (Mean Squared Error)**
   - Erro quadrático médio
   - Penaliza erros grandes

2. **RMSE (Root Mean Squared Error)**
   - Raiz do erro quadrático médio
   - Mesma unidade dos dados

3. **MAE (Mean Absolute Error)**
   - Erro absoluto médio
   - Menos sensível a outliers

4. **R² (Coefficient of Determination)**
   - Coeficiente de determinação
   - 0-1, quanto maior melhor

5. **MAPE (Mean Absolute Percentage Error)**
   - Erro percentual absoluto médio
   - Fácil interpretação

### Interpretação das Métricas

```python
# Exemplo de resultados
{
    'mse': 8.1377,      # Quanto menor, melhor
    'rmse': 2.8527,     # Erro em unidades de preço
    'mae': 2.2793,      # Erro absoluto médio
    'r2': 0.5007,       # 0.5+ é bom, 0.8+ é excelente
    'mape': 1.30        # 1.30% de erro percentual
}
```

## 🎮 Interface de Usuário

### 1. Menu Interativo (run_demo.py)

Interface de linha de comando com opções numeradas:

```
🚀 Sistema de Previsão de Séries Temporais com RNN
============================================================

📋 Menu Principal:
1. 📦 Instalar dependências
2. 📊 Carregar dados
3. 🎯 Treinar modelo
4. 📓 Executar demonstração
5. 🌐 Iniciar aplicação Streamlit
6. 🚀 Executar tudo (exceto Streamlit)
0. ❌ Sair
```

### 2. Interface Web (Streamlit)

Aplicação web interativa com:
- **Sidebar**: Configurações e parâmetros
- **Gráficos**: Visualizações interativas
- **Métricas**: Resultados em tempo real
- **Previsões**: Gráficos futuros

### 3. Jupyter Notebook

Para análise detalhada:
- Exploração de dados
- Experimentação com modelos
- Visualizações customizadas
- Relatórios completos

## 🔧 Configuração Avançada

### Personalização de Modelos

```python
# Exemplo de configuração customizada
from src.trainer import TimeSeriesTrainer

trainer = TimeSeriesTrainer(
    model_type='lstm',
    input_size=1,
    hidden_size=100,      # Camadas ocultas maiores
    num_layers=3,         # Mais camadas
    output_size=1,
    dropout=0.3           # Mais dropout
)
```

### Configuração de Dados

```python
# Exemplo de carregamento customizado
from src.data_loader import TimeSeriesDataLoader

loader = TimeSeriesDataLoader()
data = loader.download_stock_data(
    symbol='TSLA',        # Símbolo da ação
    period='2y'           # Período de dados
)

X, y = loader.prepare_sequences(
    data, 
    sequence_length=90    # Sequência mais longa
)
```

## 📊 Visualizações

### Gráficos Gerados

1. **Preços Históricos**
   - Linha temporal dos preços
   - Volume de negociação
   - Indicadores técnicos

2. **Predições vs Realidade**
   - Comparação lado a lado
   - Erros de predição
   - Intervalos de confiança

3. **Histórico de Treinamento**
   - Loss de treino e validação
   - Métricas por época
   - Early stopping

4. **Previsões Futuras**
   - Projeções para próximos dias
   - Tendências identificadas
   - Análise de volatilidade

### Exemplo de Gráfico

```python
# Código para gerar visualização
from src.utils import plot_predictions

plot_predictions(
    actuals=test_actuals,
    predictions=test_predictions,
    symbol='AAPL',
    model_type='LSTM'
)
```

## 🚀 Casos de Uso

### 1. Análise Exploratória
```bash
python run_demo.py
# Opção 2: Carregar dados
# Opção 4: Executar demonstração
```

### 2. Treinamento de Modelo
```bash
python run_demo.py
# Opção 3: Treinar modelo
# Configure parâmetros no menu
```

### 3. Interface Web Interativa
```bash
python run_demo.py
# Opção 5: Iniciar aplicação Streamlit
# Acesse http://localhost:8501
```

### 4. Análise Detalhada
```bash
jupyter notebook notebooks/analise_series_temporais.ipynb
# Execute todas as células
```

## 🔍 Solução de Problemas

### Problemas Comuns

#### 1. Erro de API Key
```bash
# Solução: Configure API keys
python setup_api_keys.py
```

#### 2. Dados Insuficientes
```bash
# Solução: Use dados sintéticos
# O sistema faz isso automaticamente
```

#### 3. Erro de Dependências
```bash
# Solução: Reinstale dependências
pip install -r requirements.txt
```

#### 4. Erro de Memória
```bash
# Solução: Reduza batch_size ou sequence_length
# Configure no menu ou código
```

### Logs e Debug

O sistema fornece logs detalhados:
- Tentativas de download de dados
- Progresso do treinamento
- Métricas de performance
- Erros e avisos

## 📚 Documentação Adicional

### Arquivos de Documentação

- **GUIA_API_KEYS.md**: Guia completo para APIs
- **notebooks/**: Exemplos e tutoriais
- **src/**: Código comentado

### Exemplos de Código

```python
# Exemplo completo de uso
from src.data_loader import TimeSeriesDataLoader
from src.trainer import TimeSeriesTrainer

# 1. Carregar dados
loader = TimeSeriesDataLoader()
data = loader.download_stock_data('AAPL', '1y')

# 2. Preparar dados
X, y = loader.prepare_sequences(data)
X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X, y)

# 3. Treinar modelo
trainer = TimeSeriesTrainer('lstm', 1, 50, 2, 1, 0.2)
train_losses, val_losses, train_metrics, val_metrics = trainer.train(
    train_loader, val_loader, epochs=50, learning_rate=0.001
)

# 4. Avaliar modelo
predictions, actuals, metrics = trainer.evaluate(X_test, y_test, loader.scaler)
print(f"R² Score: {metrics['r2']:.4f}")
```

## 🤝 Contribuição

### Como Contribuir

1. **Fork** o repositório
2. **Clone** seu fork
3. **Crie** uma branch para sua feature
4. **Implemente** suas mudanças
5. **Teste** o sistema
6. **Commit** suas mudanças
7. **Push** para sua branch
8. **Abra** um Pull Request

### Áreas para Melhoria

- Novos modelos de ML
- Mais fontes de dados
- Interface mais avançada
- Otimizações de performance
- Documentação adicional

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo LICENSE para detalhes.

## 🙏 Agradecimentos

- **APIs Financeiras**: Alpha Vantage, Polygon.io, Finnhub, etc.
- **Bibliotecas**: PyTorch, Pandas, NumPy, Streamlit, Plotly
- **Comunidade**: Contribuidores e usuários

## 📞 Suporte

Para suporte e dúvidas:
- Abra uma **Issue** no GitHub
- Consulte a **documentação**
- Execute os **testes** incluídos
- Use o **menu de ajuda** no sistema

---

**🎉 Agora você tem um sistema completo de previsão de séries temporais!**

**💡 Dica**: Comece com o menu interativo (`python run_demo.py`) para explorar todas as funcionalidades.
