#!/usr/bin/env python3
"""
Script para testar o download de dados do Yahoo Finance
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def test_yfinance_download():
    """Testa diferentes símbolos e períodos"""
    
    # Símbolos para testar
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'PETR4.SA', 'VALE3.SA']
    periods = ['1y', '6mo', '3mo', '1mo']
    
    print("🧪 Testando download de dados do Yahoo Finance")
    print("=" * 60)
    
    successful_downloads = []
    
    for symbol in symbols:
        print(f"\n📊 Testando símbolo: {symbol}")
        print("-" * 40)
        
        for period in periods:
            try:
                print(f"  Tentando período: {period}...")
                
                # Baixar dados
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                
                if not data.empty and len(data) > 5:
                    print(f"  ✅ Sucesso! {len(data)} registros baixados")
                    print(f"     Período: {data.index[0].strftime('%Y-%m-%d')} a {data.index[-1].strftime('%Y-%m-%d')}")
                    print(f"     Preço atual: ${data['Close'].iloc[-1]:.2f}")
                    
                    successful_downloads.append({
                        'symbol': symbol,
                        'period': period,
                        'records': len(data),
                        'start_date': data.index[0],
                        'end_date': data.index[-1],
                        'current_price': data['Close'].iloc[-1]
                    })
                    break  # Se conseguiu baixar, não precisa tentar outros períodos
                else:
                    print(f"  ⚠️ Dados insuficientes ({len(data)} registros)")
                    
            except Exception as e:
                print(f"  ❌ Erro: {str(e)}")
    
    # Resumo dos resultados
    print(f"\n📋 RESUMO DOS TESTES")
    print("=" * 60)
    
    if successful_downloads:
        print(f"✅ {len(successful_downloads)} símbolos baixados com sucesso:")
        for download in successful_downloads:
            print(f"  • {download['symbol']} ({download['period']}): {download['records']} registros")
            print(f"    Preço atual: ${download['current_price']:.2f}")
    else:
        print("❌ Nenhum símbolo foi baixado com sucesso")
        print("💡 Possíveis causas:")
        print("   - Problemas de conectividade com a internet")
        print("   - Yahoo Finance temporariamente indisponível")
        print("   - Símbolos inválidos ou deslistados")
    
    return successful_downloads

def test_alternative_symbols():
    """Testa símbolos alternativos"""
    
    print(f"\n🔄 Testando símbolos alternativos")
    print("=" * 60)
    
    # Testar diferentes variações de símbolos
    test_cases = [
        ('AAPL', ['AAPL', 'AAPL.US', 'AAPL.O', 'AAPL.SA']),
        ('GOOGL', ['GOOGL', 'GOOG', 'GOOGL.US', 'GOOGL.SA']),
        ('MSFT', ['MSFT', 'MSFT.US', 'MSFT.O', 'MSFT.SA']),
        ('PETR4', ['PETR4.SA', 'PETR4', 'PETR4.BA']),
        ('VALE3', ['VALE3.SA', 'VALE3', 'VALE3.BA'])
    ]
    
    for base_symbol, variations in test_cases:
        print(f"\n📊 Testando variações de {base_symbol}:")
        
        for variation in variations:
            try:
                ticker = yf.Ticker(variation)
                data = ticker.history(period='1mo')
                
                if not data.empty and len(data) > 5:
                    print(f"  ✅ {variation}: {len(data)} registros")
                    break
                else:
                    print(f"  ❌ {variation}: dados insuficientes")
                    
            except Exception as e:
                print(f"  ❌ {variation}: erro - {str(e)}")

if __name__ == "__main__":
    # Testar download básico
    successful = test_yfinance_download()
    
    # Testar símbolos alternativos
    test_alternative_symbols()
    
    print(f"\n🎯 CONCLUSÃO")
    print("=" * 60)
    
    if successful:
        print("✅ Yahoo Finance está funcionando!")
        print("💡 Recomendação: Use os símbolos que funcionaram")
    else:
        print("❌ Yahoo Finance não está funcionando")
        print("💡 Recomendação: Use dados sintéticos para demonstração")
    
    print("\n🔧 Para usar dados sintéticos, execute:")
    print("   python quick_start.py")
    print("   ou")
    print("   python run_demo.py")
