#!/usr/bin/env python3
"""
Script para testar múltiplas fontes de dados financeiros
"""

import sys
import os
sys.path.append('src')

from data_loader import TimeSeriesDataLoader
import yfinance as yf
import requests
import json

def test_yahoo_finance():
    """Testa Yahoo Finance"""
    print("🔍 Testando Yahoo Finance...")
    
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1mo')
            
            if not data.empty and len(data) > 5:
                print(f"  ✅ {symbol}: {len(data)} registros")
                print(f"     Último preço: ${data['Close'].iloc[-1]:.2f}")
            else:
                print(f"  ❌ {symbol}: dados insuficientes")
                
        except Exception as e:
            print(f"  ❌ {symbol}: {str(e)}")

def test_alpha_vantage():
    """Testa Alpha Vantage"""
    print("\n🔍 Testando Alpha Vantage...")
    
    # API key gratuita (você pode se registrar em https://www.alphavantage.co/)
    api_key = "demo"  # Substitua pela sua API key
    
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    for symbol in symbols:
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': api_key,
                'outputsize': 'compact'
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'Time Series (Daily)' in data:
                print(f"  ✅ {symbol}: dados disponíveis")
            else:
                print(f"  ❌ {symbol}: {data.get('Note', 'Erro desconhecido')}")
                
        except Exception as e:
            print(f"  ❌ {symbol}: {str(e)}")

def test_polygon():
    """Testa Polygon.io"""
    print("\n🔍 Testando Polygon.io...")
    
    # API key gratuita (você pode se registrar em https://polygon.io/)
    api_key = "demo"  # Substitua pela sua API key
    
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    for symbol in symbols:
        try:
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/2024-01-01/2024-01-31"
            params = {'apiKey': api_key}
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get('status') == 'OK':
                print(f"  ✅ {symbol}: {len(data.get('results', []))} registros")
            else:
                print(f"  ❌ {symbol}: {data.get('error', 'Erro desconhecido')}")
                
        except Exception as e:
            print(f"  ❌ {symbol}: {str(e)}")

def test_finnhub():
    """Testa Finnhub"""
    print("\n🔍 Testando Finnhub...")
    
    # API key gratuita (você pode se registrar em https://finnhub.io/)
    api_key = "demo"  # Substitua pela sua API key
    
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    for symbol in symbols:
        try:
            url = "https://finnhub.io/api/v1/stock/candle"
            params = {
                'symbol': symbol,
                'resolution': 'D',
                'from': 1704067200,  # 2024-01-01
                'to': 1706745600,    # 2024-01-31
                'token': api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get('s') == 'ok':
                print(f"  ✅ {symbol}: {len(data.get('t', []))} registros")
            else:
                print(f"  ❌ {symbol}: {data.get('error', 'Erro desconhecido')}")
                
        except Exception as e:
            print(f"  ❌ {symbol}: {str(e)}")

def test_data_loader():
    """Testa o carregador de dados integrado"""
    print("\n🔍 Testando DataLoader integrado...")
    
    loader = TimeSeriesDataLoader()
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    for symbol in symbols:
        try:
            print(f"\n📊 Testando {symbol}...")
            data = loader.download_stock_data(symbol, period='1mo')
            
            if data is not None and not data.empty:
                print(f"  ✅ {symbol}: {len(data)} registros baixados")
                print(f"     Período: {data.index[0].strftime('%Y-%m-%d')} a {data.index[-1].strftime('%Y-%m-%d')}")
                print(f"     Último preço: ${data['Close'].iloc[-1]:.2f}")
            else:
                print(f"  ❌ {symbol}: dados insuficientes")
                
        except Exception as e:
            print(f"  ❌ {symbol}: {str(e)}")

def setup_api_keys():
    """Guia para configurar API keys"""
    print("\n🔑 CONFIGURAÇÃO DE API KEYS")
    print("=" * 60)
    print("Para obter dados reais, você precisa se registrar nas seguintes APIs:")
    print()
    print("1. Alpha Vantage (https://www.alphavantage.co/)")
    print("   - Gratuito: 500 requisições/dia")
    print("   - Registre-se e obtenha sua API key")
    print()
    print("2. Polygon.io (https://polygon.io/)")
    print("   - Gratuito: 5 requisições/minuto")
    print("   - Registre-se e obtenha sua API key")
    print()
    print("3. Finnhub (https://finnhub.io/)")
    print("   - Gratuito: 60 requisições/minuto")
    print("   - Registre-se e obtenha sua API key")
    print()
    print("📝 Para usar suas API keys:")
    print("1. Edite o arquivo src/data_loader.py")
    print("2. Substitua 'demo' pelas suas API keys reais")
    print("3. Execute novamente este script")

def main():
    """Função principal"""
    print("🧪 TESTE DE FONTES DE DADOS FINANCEIROS")
    print("=" * 60)
    
    # Testar cada fonte individualmente
    test_yahoo_finance()
    test_alpha_vantage()
    test_polygon()
    test_finnhub()
    
    # Testar o data loader integrado
    test_data_loader()
    
    # Mostrar guia de configuração
    setup_api_keys()
    
    print("\n🎯 CONCLUSÃO")
    print("=" * 60)
    print("✅ Yahoo Finance: Geralmente funciona sem API key")
    print("⚠️ Outras APIs: Requerem registro e API key")
    print("💡 Recomendação: Use Yahoo Finance para testes rápidos")
    print("🚀 Para produção: Configure API keys das outras fontes")

if __name__ == "__main__":
    main()
