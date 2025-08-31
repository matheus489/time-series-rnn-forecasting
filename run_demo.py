"""
Script principal para executar a demonstração completa do sistema de previsão de séries temporais
"""

import os
import sys
import subprocess
import argparse

def install_requirements():
    """Instala as dependências necessárias"""
    print("📦 Instalando dependências...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependências instaladas com sucesso!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Erro ao instalar dependências")
        return False

def run_data_loader():
    """Executa o carregador de dados"""
    print("\n📊 Executando carregador de dados...")
    try:
        subprocess.check_call([sys.executable, "src/data_loader.py"])
        print("✅ Dados carregados com sucesso!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Erro ao carregar dados")
        return False

def run_trainer():
    """Executa o treinamento do modelo"""
    print("\n🎯 Executando treinamento do modelo...")
    try:
        subprocess.check_call([sys.executable, "src/trainer.py"])
        print("✅ Modelo treinado com sucesso!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Erro ao treinar modelo")
        return False

def run_streamlit():
    """Executa a aplicação Streamlit"""
    print("\n🌐 Iniciando aplicação Streamlit...")
    print("📱 A aplicação será aberta no seu navegador")
    print("🔗 URL: http://localhost:8501")
    print("⏹️  Pressione Ctrl+C para parar a aplicação")
    
    try:
        subprocess.check_call([sys.executable, "-m", "streamlit", "run", "src/streamlit_app.py"])
    except subprocess.CalledProcessError:
        print("❌ Erro ao iniciar aplicação Streamlit")
    except KeyboardInterrupt:
        print("\n👋 Aplicação Streamlit encerrada")

def run_notebook_demo():
    """Executa a demonstração do notebook"""
    print("\n📓 Executando demonstração do notebook...")
    try:
        subprocess.check_call([sys.executable, "notebooks/demo_analysis.py"])
        print("✅ Demonstração executada com sucesso!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Erro ao executar demonstração")
        return False

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description="Sistema de Previsão de Séries Temporais com RNN")
    parser.add_argument("--install", action="store_true", help="Instalar dependências")
    parser.add_argument("--data", action="store_true", help="Carregar dados")
    parser.add_argument("--train", action="store_true", help="Treinar modelo")
    parser.add_argument("--streamlit", action="store_true", help="Executar aplicação Streamlit")
    parser.add_argument("--demo", action="store_true", help="Executar demonstração")
    parser.add_argument("--all", action="store_true", help="Executar tudo (exceto Streamlit)")
    
    args = parser.parse_args()
    
    print("🚀 Sistema de Previsão de Séries Temporais com RNN")
    print("=" * 60)
    
    # Verificar se estamos no diretório correto
    if not os.path.exists("src"):
        print("❌ Erro: Execute este script no diretório raiz do projeto")
        return
    
    success = True
    
    # Instalar dependências se solicitado
    if args.install or args.all:
        success = install_requirements() and success
    
    # Carregar dados se solicitado
    if args.data or args.all:
        success = run_data_loader() and success
    
    # Treinar modelo se solicitado
    if args.train or args.all:
        success = run_trainer() and success
    
    # Executar demonstração se solicitado
    if args.demo or args.all:
        success = run_notebook_demo() and success
    
    # Executar Streamlit se solicitado
    if args.streamlit:
        run_streamlit()
        return
    
    # Se nenhum argumento foi fornecido, mostrar menu interativo
    if not any([args.install, args.data, args.train, args.streamlit, args.demo, args.all]):
        show_interactive_menu()
    
    if success:
        print("\n✅ Operação concluída com sucesso!")
    else:
        print("\n❌ Algumas operações falharam")

def show_interactive_menu():
    """Mostra menu interativo"""
    while True:
        print("\n📋 Menu Principal:")
        print("1. 📦 Instalar dependências")
        print("2. 📊 Carregar dados")
        print("3. 🎯 Treinar modelo")
        print("4. 📓 Executar demonstração")
        print("5. 🌐 Iniciar aplicação Streamlit")
        print("6. 🚀 Executar tudo (exceto Streamlit)")
        print("0. ❌ Sair")
        
        choice = input("\nEscolha uma opção: ").strip()
        
        if choice == "1":
            install_requirements()
        elif choice == "2":
            run_data_loader()
        elif choice == "3":
            run_trainer()
        elif choice == "4":
            run_notebook_demo()
        elif choice == "5":
            run_streamlit()
            break  # Sair após Streamlit
        elif choice == "6":
            install_requirements()
            run_data_loader()
            run_trainer()
            run_notebook_demo()
        elif choice == "0":
            print("👋 Até logo!")
            break
        else:
            print("❌ Opção inválida. Tente novamente.")

if __name__ == "__main__":
    main()

