# 📈 Resumo Executivo - Sistema de Previsão de Séries Temporais

## 🎯 O que é?

Um sistema completo de **Machine Learning** para prever preços de ações usando **Redes Neurais Recorrentes (RNN)**. Funciona com dados reais de múltiplas APIs financeiras e tem interface amigável.

## 🚀 Como usar (3 passos simples)

### 1. Instalar
```bash
pip install -r requirements.txt
```

### 2. Configurar APIs (opcional)
```bash
python setup_api_keys.py
# Ou use dados sintéticos automaticamente
```

### 3. Executar
```bash
python run_demo.py
# Escolha a opção desejada no menu
```

## 📊 O que o sistema faz?

### ✅ Funcionalidades Principais
- **📥 Baixa dados** de 6 APIs diferentes (Yahoo Finance, Alpha Vantage, etc.)
- **🧠 Treina modelos** LSTM, GRU e RNN
- **📈 Faz previsões** de preços futuros
- **📊 Gera gráficos** interativos
- **📋 Calcula métricas** de performance (R², RMSE, MAPE)

### 🔄 Sistema Robusto
- **Fallback automático**: Se APIs falharem, usa dados sintéticos
- **Múltiplas fontes**: Tenta 6 APIs diferentes
- **Configuração flexível**: Parâmetros ajustáveis

## 🎮 Interfaces Disponíveis

| Interface | Comando | Uso |
|-----------|---------|-----|
| **Menu Interativo** | `python run_demo.py` | Principal - todas as opções |
| **Início Rápido** | `python quick_start.py` | Demonstração automática |
| **Interface Web** | Streamlit | Visualizações interativas |
| **Jupyter Notebook** | `jupyter notebook` | Análise detalhada |

## 📈 Modelos de ML

| Modelo | Vantagem | Uso Recomendado |
|--------|----------|-----------------|
| **LSTM** | Melhor para séries longas | Padrão para ações |
| **GRU** | Mais rápido que LSTM | Alternativa eficiente |
| **RNN** | Simples e rápido | Comparação/baseline |

## 🔗 APIs Suportadas

| API | Limite Gratuito | Facilidade |
|-----|----------------|------------|
| **Alpha Vantage** | 500 req/dia | ⭐⭐⭐⭐⭐ (Recomendado) |
| **Yahoo Finance** | Ilimitado | ⭐⭐⭐⭐⭐ (Sem API key) |
| **Polygon.io** | 5 req/min | ⭐⭐⭐⭐ |
| **Finnhub** | 60 req/min | ⭐⭐⭐⭐ |

## 📊 Métricas de Performance

| Métrica | Significado | Bom Valor |
|---------|-------------|-----------|
| **R²** | Qualidade da predição | > 0.5 |
| **RMSE** | Erro em $ | Quanto menor melhor |
| **MAPE** | Erro percentual | < 5% |
| **MAE** | Erro absoluto | Quanto menor melhor |

## 🎯 Casos de Uso

### 👨‍💻 Para Desenvolvedores
- Experimentar com RNNs
- Aprender séries temporais
- Testar diferentes modelos

### 📊 Para Analistas
- Análise de ações
- Previsões de mercado
- Relatórios financeiros

### 🎓 Para Estudantes
- Projetos acadêmicos
- Pesquisa em ML
- Portfólio de projetos

## 🔧 Configuração Rápida

### Dados Reais (Recomendado)
```bash
# 1. Configure API keys
python setup_api_keys.py

# 2. Teste as APIs
python test_data_sources.py

# 3. Execute o sistema
python run_demo.py
```

### Dados Sintéticos (Fallback)
```bash
# Funciona automaticamente se APIs falharem
python quick_start.py
```

## 📁 Estrutura de Arquivos

```
📁 Projeto/
├── 📄 run_demo.py          # 🎮 Menu principal
├── 📄 quick_start.py       # ⚡ Início rápido
├── 📄 setup_api_keys.py    # 🔑 Configurar APIs
├── 📁 src/                 # 🧠 Código fonte
├── 📁 notebooks/           # 📓 Análises
└── 📁 data/                # 📊 Dados baixados
```

## 🚀 Exemplo de Uso Rápido

```bash
# 1. Instalar
pip install -r requirements.txt

# 2. Executar
python run_demo.py

# 3. Escolher opção 6: "Executar tudo"
# 4. Aguardar treinamento
# 5. Ver resultados e gráficos
```

## 💡 Dicas Importantes

### ✅ Para Iniciantes
1. **Comece com dados sintéticos** (funciona sem configuração)
2. **Use o menu interativo** (`python run_demo.py`)
3. **Configure Alpha Vantage** (mais fácil)

### ⚡ Para Uso Avançado
1. **Configure múltiplas APIs** para redundância
2. **Ajuste parâmetros** dos modelos
3. **Use Jupyter notebooks** para análise detalhada

### 🔧 Para Desenvolvimento
1. **Modifique modelos** em `src/model.py`
2. **Adicione APIs** em `src/data_loader.py`
3. **Customize visualizações** em `src/utils.py`

## 📞 Suporte

- **Problemas**: Execute `python test_data_sources.py`
- **Configuração**: Use `python setup_api_keys.py`
- **Documentação**: Veja `README.md` completo
- **Exemplos**: Explore `notebooks/`

---

**🎉 Pronto para usar! O sistema é robusto e funciona mesmo sem configuração.**

**💡 Comece com**: `python run_demo.py` → Opção 6
