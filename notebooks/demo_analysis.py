"""
Demonstração de Análise de Séries Temporais com RNN
Este script pode ser executado como alternativa ao Jupyter Notebook
"""

import sys
import os

# Adicionar o diretório src ao path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(current_dir), 'src')
sys.path.append(src_dir)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configuração de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Importar módulos locais
from data_loader import TimeSeriesDataLoader
from model import create_model
from trainer import TimeSeriesTrainer
from utils import (
    plot_training_history, 
    plot_predictions, 
    plot_interactive_predictions,
    plot_future_predictions,
    create_performance_report,
    plot_model_comparison,
    analyze_residuals
)

def main():
    """Função principal da demonstração"""
    print("🚀 Iniciando Demonstração de Análise de Séries Temporais com RNN")
    print("=" * 70)
    
    # 1. Carregamento de dados
    print("\n1. 📊 Carregamento e Análise de Dados")
    print("-" * 40)
    
    symbol = 'AAPL'
    period = '2y'
    
    # Inicializar carregador
    loader = TimeSeriesDataLoader()
    
    # Baixar dados
    print(f"Baixando dados para {symbol}...")
    try:
        data = loader.download_stock_data(symbol, period)
        print(f"✅ Dados reais baixados com sucesso!")
    except Exception as e:
        print(f"⚠️ Erro ao baixar dados reais: {e}")
        print("🔄 Usando dados sintéticos para demonstração...")
        data = loader.generate_synthetic_data(symbol, period)
    
    print(f"Dados baixados com sucesso!")
    print(f"Período: {data.index[0].strftime('%Y-%m-%d')} a {data.index[-1].strftime('%Y-%m-%d')}")
    print(f"Total de registros: {len(data)}")
    
    # 2. Preparação dos dados
    print("\n2. 🔧 Preparação dos Dados")
    print("-" * 40)
    
    sequence_length = 60
    X, y = loader.prepare_sequences(data, sequence_length=sequence_length)
    
    print(f"Shape dos dados de entrada: {X.shape}")
    print(f"Shape dos dados de saída: {y.shape}")
    print(f"Cada sequência tem {sequence_length} dias de histórico")
    
    # Dividir dados
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X, y)
    
    print(f"Divisão dos dados:")
    print(f"Treino: {X_train.shape[0]} amostras ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Validação: {X_val.shape[0]} amostras ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"Teste: {X_test.shape[0]} amostras ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # 3. Treinamento de modelos
    print("\n3. 🎯 Treinamento de Modelos")
    print("-" * 40)
    
    # Configurações
    model_configs = {
        'lstm': {'hidden_size': 50, 'num_layers': 2, 'dropout': 0.2},
        'gru': {'hidden_size': 50, 'num_layers': 2, 'dropout': 0.2}
    }
    
    train_config = {
        'epochs': 30,  # Reduzido para demonstração
        'batch_size': 32,
        'learning_rate': 0.001
    }
    
    results = {}
    
    # Treinar modelos
    for model_type, config in model_configs.items():
        print(f"\nTreinando modelo {model_type.upper()}...")
        
        # Criar trainer
        trainer = TimeSeriesTrainer(
            model_type=model_type,
            input_size=1,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=1,
            dropout=config['dropout']
        )
        
        # Preparar dataloaders
        train_loader, val_loader = trainer.prepare_data_loaders(
            X_train, y_train, X_val, y_val, batch_size=train_config['batch_size']
        )
        
        # Treinar modelo
        train_losses, val_losses, train_metrics, val_metrics = trainer.train(
            train_loader, val_loader,
            epochs=train_config['epochs'],
            learning_rate=train_config['learning_rate'],
            patience=10
        )
        
        # Avaliar modelo
        predictions, actuals, test_metrics = trainer.evaluate(
            X_test, y_test, loader.scaler
        )
        
        # Armazenar resultados
        results[model_type] = {
            'trainer': trainer,
            'predictions': predictions,
            'actuals': actuals,
            'test_metrics': test_metrics,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
        
        print(f"✅ Modelo {model_type.upper()} treinado!")
    
    # 4. Comparação de performance
    print("\n4. 📈 Comparação de Performance")
    print("-" * 40)
    
    comparison_data = {}
    for model_type, result in results.items():
        comparison_data[model_type] = result['test_metrics']
    
    comparison_df = pd.DataFrame(comparison_data).T
    print("Comparação de Performance dos Modelos:")
    print(comparison_df)
    
    # 5. Análise do melhor modelo
    print("\n5. 🏆 Análise do Melhor Modelo")
    print("-" * 40)
    
    # Identificar melhor modelo
    best_model = max(comparison_data.keys(), key=lambda x: comparison_data[x]['r2'])
    print(f"Melhor modelo: {best_model.upper()} (R² = {comparison_data[best_model]['r2']:.4f})")
    
    best_result = results[best_model]
    
    # Plotar resultados do melhor modelo
    print("Gerando visualizações...")
    plot_training_history(
        best_result['train_losses'],
        best_result['val_losses'],
        best_result['train_metrics'],
        best_result['val_metrics'],
        symbol,
        best_model
    )
    
    plot_predictions(
        best_result['actuals'],
        best_result['predictions'],
        symbol,
        best_model
    )
    
    # 6. Previsões futuras
    print("\n6. 🔮 Previsões Futuras")
    print("-" * 40)
    
    days_ahead = 30
    last_sequence = X_test[-1]
    
    future_predictions = best_result['trainer'].predict_future(
        last_sequence, loader.scaler, days_ahead
    )
    
    print(f"Previsões para os próximos {days_ahead} dias:")
    for i, pred in enumerate(future_predictions[:10]):  # Mostrar apenas os primeiros 10
        print(f"Dia {i+1:2d}: ${pred:.2f}")
    
    # Plotar previsões futuras
    plot_future_predictions(
        data,
        future_predictions,
        symbol,
        best_model,
        days_ahead
    )
    
    # 7. Relatório final
    print("\n7. 📋 Relatório Final")
    print("-" * 40)
    
    report = create_performance_report(
        best_result['test_metrics'],
        symbol,
        best_model
    )
    
    print("RESUMO DOS RESULTADOS:")
    print("-" * 40)
    for model_type, result in results.items():
        metrics = result['test_metrics']
        print(f"{model_type.upper()}:")
        print(f"  R²: {metrics['r2']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        print()
    
    print(f"🎯 Melhor modelo: {best_model.upper()}")
    print(f"📈 R² Score: {comparison_data[best_model]['r2']:.4f}")
    print(f"📊 Precisão: {100 - comparison_data[best_model]['mape']:.2f}%")
    
    print("\n✅ Demonstração concluída com sucesso!")
    print("📁 Verifique as pastas 'plots/', 'models/' e 'reports/' para os resultados salvos.")

if __name__ == "__main__":
    main()

