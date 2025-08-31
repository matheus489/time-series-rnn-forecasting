#!/usr/bin/env python3
"""
Script para configurar API keys para fontes de dados financeiros
"""

import os
import json

def setup_api_keys():
    """Configura API keys interativamente"""
    
    print("🔑 CONFIGURAÇÃO DE API KEYS")
    print("=" * 60)
    print("Este script irá ajudá-lo a configurar suas API keys para obter dados reais.")
    print("As API keys serão salvas em um arquivo de configuração local.")
    print()
    
    # Dicionário para armazenar as API keys
    api_keys = {}
    
    # Alpha Vantage
    print("1️⃣ ALPHA VANTAGE")
    print("   - Site: https://www.alphavantage.co/")
    print("   - Gratuito: 500 requisições/dia")
    print("   - Registre-se e obtenha sua API key gratuita")
    print()
    
    alpha_key = input("Digite sua Alpha Vantage API key (ou pressione Enter para pular): ").strip()
    if alpha_key:
        api_keys['alpha_vantage'] = alpha_key
        print("✅ Alpha Vantage configurado!")
    else:
        print("⏭️ Alpha Vantage pulado")
    
    print()
    
    # Polygon.io
    print("2️⃣ POLYGON.IO")
    print("   - Site: https://polygon.io/")
    print("   - Gratuito: 5 requisições/minuto")
    print("   - Registre-se e obtenha sua API key gratuita")
    print()
    
    polygon_key = input("Digite sua Polygon.io API key (ou pressione Enter para pular): ").strip()
    if polygon_key:
        api_keys['polygon'] = polygon_key
        print("✅ Polygon.io configurado!")
    else:
        print("⏭️ Polygon.io pulado")
    
    print()
    
    # Finnhub
    print("3️⃣ FINNHUB")
    print("   - Site: https://finnhub.io/")
    print("   - Gratuito: 60 requisições/minuto")
    print("   - Registre-se e obtenha sua API key gratuita")
    print()
    
    finnhub_key = input("Digite sua Finnhub API key (ou pressione Enter para pular): ").strip()
    if finnhub_key:
        api_keys['finnhub'] = finnhub_key
        print("✅ Finnhub configurado!")
    else:
        print("⏭️ Finnhub pulado")
    
    print()
    
    # Salvar configuração
    if api_keys:
        config_file = 'api_config.json'
        with open(config_file, 'w') as f:
            json.dump(api_keys, f, indent=2)
        
        print(f"✅ Configuração salva em: {config_file}")
        print()
        print("📝 Para usar estas API keys:")
        print("1. O sistema irá carregá-las automaticamente")
        print("2. Se preferir, você pode editar o arquivo src/data_loader.py")
        print("3. Execute python test_data_sources.py para testar")
    else:
        print("⚠️ Nenhuma API key foi configurada")
        print("💡 O sistema usará Yahoo Finance (que não requer API key)")
    
    print()
    print("🎯 PRÓXIMOS PASSOS:")
    print("1. Execute: python test_data_sources.py")
    print("2. Execute: python quick_start.py")
    print("3. Execute: python run_demo.py")

def load_api_keys():
    """Carrega API keys do arquivo de configuração"""
    config_file = 'api_config.json'
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Erro ao carregar configuração: {e}")
    
    return {}

def update_data_loader():
    """Atualiza o data_loader.py com as API keys configuradas"""
    api_keys = load_api_keys()
    
    if not api_keys:
        print("⚠️ Nenhuma API key configurada. Execute setup_api_keys.py primeiro.")
        return
    
    print("🔄 Atualizando data_loader.py com suas API keys...")
    
    # Ler o arquivo atual
    with open('src/data_loader.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Substituir as API keys
    for service, key in api_keys.items():
        if service == 'alpha_vantage':
            content = content.replace("api_key = \"demo\"  # Use sua própria API key", f"api_key = \"{key}\"")
        elif service == 'polygon':
            content = content.replace("api_key = \"demo\"  # Use sua própria API key", f"api_key = \"{key}\"")
        elif service == 'finnhub':
            content = content.replace("api_key = \"demo\"  # Use sua própria API key", f"api_key = \"{key}\"")
    
    # Salvar o arquivo atualizado
    with open('src/data_loader.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ data_loader.py atualizado com suas API keys!")

if __name__ == "__main__":
    print("🚀 CONFIGURADOR DE API KEYS")
    print("=" * 60)
    
    choice = input("Escolha uma opção:\n1. Configurar API keys\n2. Atualizar data_loader.py\n3. Ambas\nDigite sua escolha (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        setup_api_keys()
    
    if choice in ['2', '3']:
        update_data_loader()
    
    print("\n✅ Configuração concluída!")
