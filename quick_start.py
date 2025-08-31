"""
Início Rápido - Demonstração Simples do Sistema de Previsão de Séries Temporais
"""

import sys
import os
sys.path.append('src')

from data_loader import TimeSeriesDataLoader
from model import create_model
from trainer import TimeSeriesTrainer
from utils import plot_predictions, create_performance_report

def quick_demo():
    """Demonstração rápida do sistema"""
    print("🚀 Início Rápido - Sistema de Previsão de Séries Temporais")
    print("=" * 60)
    
    # Lista de símbolos para tentar (em ordem de preferência)
    symbols_to_try = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    
    # 1. Carregar dados
    print("\n1. 📊 Carregando dados de ações...")
    loader = TimeSeriesDataLoader()
    
    data = None
    successful_symbol = None
    
    for symbol in symbols_to_try:
        try:
            print(f"Tentando baixar dados para {symbol}...")
            data = loader.download_stock_data(symbol, period='1y')
            successful_symbol = symbol
            print(f"✅ Dados carregados para {symbol}: {len(data)} registros")
            break
        except Exception as e:
            print(f"❌ Falha ao baixar {symbol}: {str(e)}")
            continue
    
    if data is None:
        print("❌ Não foi possível baixar dados para nenhum símbolo.")
        print("🔄 Usando dados sintéticos para demonstração...")
        data = loader.generate_synthetic_data('DEMO', period='1y')
        successful_symbol = 'DEMO'
    
    # 2. Preparar dados
    print(f"\n2. 🔧 Preparando dados para {successful_symbol}...")
    X, y = loader.prepare_sequences(data, sequence_length=30)  # Sequência menor para rapidez
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X, y)
    print(f"✅ Dados preparados: {len(X_train)} amostras de treino")
    
    # 3. Criar e treinar modelo
    print("\n3. 🎯 Treinando modelo LSTM...")
    trainer = TimeSeriesTrainer(
        model_type='lstm',
        input_size=1,
        hidden_size=30,  # Menor para rapidez
        num_layers=1,
        output_size=1,
        dropout=0.1
    )
    
    # Preparar dataloaders
    train_loader, val_loader = trainer.prepare_data_loaders(
        X_train, y_train, X_val, y_val, batch_size=16
    )
    
    # Treinar modelo (poucas épocas para rapidez)
    train_losses, val_losses, train_metrics, val_metrics = trainer.train(
        train_loader, val_loader,
        epochs=20,  # Poucas épocas
        learning_rate=0.001,
        patience=5
    )
    
    # 4. Avaliar modelo
    print("\n4. 📈 Avaliando modelo...")
    predictions, actuals, test_metrics = trainer.evaluate(
        X_test, y_test, loader.scaler
    )
    
    # 5. Mostrar resultados
    print("\n5. 📊 Resultados:")
    print(f"R² Score: {test_metrics['r2']:.4f}")
    print(f"RMSE: {test_metrics['rmse']:.4f}")
    print(f"MAPE: {test_metrics['mape']:.2f}%")
    
    # 6. Gerar visualização
    print("\n6. 📈 Gerando gráfico de predições...")
    plot_predictions(actuals, predictions, successful_symbol, 'lstm')
    
    # 7. Fazer previsão futura
    print("\n7. 🔮 Fazendo previsão para os próximos 7 dias...")
    last_sequence = X_test[-1]
    future_predictions = trainer.predict_future(last_sequence, loader.scaler, steps=7)
    
    print("Previsões:")
    for i, pred in enumerate(future_predictions):
        print(f"Dia {i+1}: ${pred:.2f}")
    
    print("\n✅ Demonstração concluída!")
    print("📁 Verifique a pasta 'plots/' para ver os gráficos gerados")

if __name__ == "__main__":
    quick_demo()

