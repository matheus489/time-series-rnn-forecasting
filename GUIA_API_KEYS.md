# 🔑 Guia Completo para Configurar API Keys e Obter Dados Reais

## 📋 Visão Geral

Este sistema suporta múltiplas fontes de dados financeiros. Atualmente, quando as APIs não estão configuradas, o sistema usa dados sintéticos para demonstração. Para obter dados reais, você pode configurar uma ou mais das seguintes APIs:

## 🚀 APIs Disponíveis

### 1. Alpha Vantage (Recomendado para iniciantes)
- **Site**: https://www.alphavantage.co/
- **Gratuito**: 500 requisições/dia
- **Registro**: Gratuito, sem cartão de crédito
- **Facilidade**: ⭐⭐⭐⭐⭐

### 2. Polygon.io
- **Site**: https://polygon.io/
- **Gratuito**: 5 requisições/minuto
- **Registro**: Gratuito, sem cartão de crédito
- **Facilidade**: ⭐⭐⭐⭐

### 3. Finnhub
- **Site**: https://finnhub.io/
- **Gratuito**: 60 requisições/minuto
- **Registro**: Gratuito, sem cartão de crédito
- **Facilidade**: ⭐⭐⭐⭐

### 4. IEX Cloud
- **Site**: https://iexcloud.io/
- **Gratuito**: 50,000 mensagens/mês
- **Registro**: Gratuito, requer cartão de crédito
- **Facilidade**: ⭐⭐⭐

### 5. Quandl
- **Site**: https://www.quandl.com/
- **Gratuito**: 50 requisições/dia
- **Registro**: Gratuito, sem cartão de crédito
- **Facilidade**: ⭐⭐⭐

## 🔧 Como Configurar

### Opção 1: Configuração Automática (Recomendada)

1. **Execute o script de configuração**:
   ```bash
   python setup_api_keys.py
   ```

2. **Siga as instruções** para cada API que desejar configurar

3. **Teste a configuração**:
   ```bash
   python test_data_sources.py
   ```

### Opção 2: Configuração Manual

1. **Registre-se nas APIs** desejadas (links acima)

2. **Crie o arquivo `api_config.json`** na raiz do projeto:
   ```json
   {
     "alpha_vantage": "SUA_API_KEY_AQUI",
     "polygon": "SUA_API_KEY_AQUI",
     "finnhub": "SUA_API_KEY_AQUI",
     "iex": "SUA_API_KEY_AQUI",
     "quandl": "SUA_API_KEY_AQUI"
   }
   ```

3. **Substitua** "SUA_API_KEY_AQUI" pelas suas chaves reais

## 📊 Testando as APIs

Após configurar as API keys, teste o sistema:

```bash
# Testar todas as fontes
python test_data_sources.py

# Executar demonstração com dados reais
python quick_start.py

# Executar interface interativa
python run_demo.py
```

## 🎯 Recomendações

### Para Iniciantes
1. **Comece com Alpha Vantage** - é a mais fácil de configurar
2. **500 requisições/dia** são suficientes para testes
3. **Registro gratuito** sem cartão de crédito

### Para Uso Intensivo
1. **Configure múltiplas APIs** para redundância
2. **Monitore os limites** de cada API
3. **Considere planos pagos** para uso comercial

### Para Produção
1. **Use APIs pagas** para maior confiabilidade
2. **Implemente cache** para reduzir requisições
3. **Monitore quotas** e implemente fallbacks

## 🔍 Solução de Problemas

### Erro: "API key inválida"
- Verifique se a API key está correta
- Confirme se a conta está ativa
- Verifique se não excedeu o limite gratuito

### Erro: "Dados insuficientes"
- Tente símbolos diferentes (AAPL, GOOGL, MSFT)
- Verifique se o símbolo existe na bolsa
- Tente períodos menores (1mo, 3mo)

### Erro: "Timeout"
- Verifique sua conexão com a internet
- Tente novamente em alguns minutos
- Use dados sintéticos temporariamente

## 📈 Exemplo de Uso com Dados Reais

Após configurar as APIs:

```python
from src.data_loader import TimeSeriesDataLoader

# Inicializar carregador
loader = TimeSeriesDataLoader()

# Baixar dados reais
data = loader.download_stock_data('AAPL', period='1y')

print(f"Dados baixados: {len(data)} registros")
print(f"Último preço: ${data['Close'].iloc[-1]:.2f}")
```

## 🎉 Benefícios dos Dados Reais

1. **Precisão**: Dados reais do mercado
2. **Atualização**: Preços em tempo real
3. **Variedade**: Múltiplas ações disponíveis
4. **Histórico**: Dados históricos completos
5. **Confiabilidade**: Fontes oficiais de dados

## 📞 Suporte

Se você encontrar problemas:

1. **Verifique os logs** de erro
2. **Teste cada API** individualmente
3. **Consulte a documentação** da API
4. **Use dados sintéticos** como fallback

## 🚀 Próximos Passos

1. **Configure pelo menos uma API**
2. **Teste com dados reais**
3. **Explore diferentes ações**
4. **Compare resultados** entre APIs
5. **Implemente cache** para otimização

---

**💡 Dica**: Mesmo sem API keys, o sistema funciona perfeitamente com dados sintéticos para demonstração e aprendizado!
