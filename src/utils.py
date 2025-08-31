"""
Utilitários para visualização e análise de séries temporais
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_training_history(train_losses, val_losses, train_metrics, val_metrics, symbol, model_type):
    """
    Plota o histórico de treinamento
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Histórico de Treinamento - {symbol} ({model_type.upper()})', fontsize=16)
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot 1: Loss
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Treino', linewidth=2)
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Validação', linewidth=2)
    axes[0, 0].set_title('Loss durante o Treinamento')
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: MSE
    train_mse = [m['mse'] for m in train_metrics]
    val_mse = [m['mse'] for m in val_metrics]
    axes[0, 1].plot(epochs, train_mse, 'b-', label='Treino', linewidth=2)
    axes[0, 1].plot(epochs, val_mse, 'r-', label='Validação', linewidth=2)
    axes[0, 1].set_title('MSE durante o Treinamento')
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: MAE
    train_mae = [m['mae'] for m in train_metrics]
    val_mae = [m['mae'] for m in val_metrics]
    axes[1, 0].plot(epochs, train_mae, 'b-', label='Treino', linewidth=2)
    axes[1, 0].plot(epochs, val_mae, 'r-', label='Validação', linewidth=2)
    axes[1, 0].set_title('MAE durante o Treinamento')
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: R²
    train_r2 = [m['r2'] for m in train_metrics]
    val_r2 = [m['r2'] for m in val_metrics]
    axes[1, 1].plot(epochs, train_r2, 'b-', label='Treino', linewidth=2)
    axes[1, 1].plot(epochs, val_r2, 'r-', label='Validação', linewidth=2)
    axes[1, 1].set_title('R² durante o Treinamento')
    axes[1, 1].set_xlabel('Época')
    axes[1, 1].set_ylabel('R²')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Salvar gráfico
    if not os.path.exists('plots'):
        os.makedirs('plots')
    plt.savefig(f'plots/{symbol}_{model_type}_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_predictions(actuals, predictions, symbol, model_type):
    """
    Plota comparação entre valores reais e predições
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    fig.suptitle(f'Comparação: Valores Reais vs Predições - {symbol} ({model_type.upper()})', fontsize=16)
    
    # Plot 1: Série temporal
    x = range(len(actuals))
    axes[0].plot(x, actuals, 'b-', label='Valores Reais', linewidth=2, alpha=0.8)
    axes[0].plot(x, predictions, 'r-', label='Predições', linewidth=2, alpha=0.8)
    axes[0].set_title('Série Temporal: Valores Reais vs Predições')
    axes[0].set_xlabel('Período')
    axes[0].set_ylabel('Preço ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot
    axes[1].scatter(actuals, predictions, alpha=0.6, color='green')
    axes[1].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', linewidth=2)
    axes[1].set_title('Scatter Plot: Valores Reais vs Predições')
    axes[1].set_xlabel('Valores Reais')
    axes[1].set_ylabel('Predições')
    axes[1].grid(True, alpha=0.3)
    
    # Adicionar R² no gráfico
    r2 = np.corrcoef(actuals, predictions)[0, 1] ** 2
    axes[1].text(0.05, 0.95, f'R² = {r2:.4f}', transform=axes[1].transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    # Criar pasta plots se não existir
    import os
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    # Salvar gráfico
    plt.savefig(f'plots/{symbol}_{model_type}_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_interactive_predictions(actuals, predictions, symbol, model_type):
    """
    Cria gráfico interativo usando Plotly
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Série Temporal: Valores Reais vs Predições', 
                       'Scatter Plot: Valores Reais vs Predições'),
        vertical_spacing=0.1
    )
    
    # Plot 1: Série temporal
    x = list(range(len(actuals)))
    fig.add_trace(
        go.Scatter(x=x, y=actuals, mode='lines', name='Valores Reais', 
                  line=dict(color='blue', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x, y=predictions, mode='lines', name='Predições', 
                  line=dict(color='red', width=2)),
        row=1, col=1
    )
    
    # Plot 2: Scatter plot
    fig.add_trace(
        go.Scatter(x=actuals, y=predictions, mode='markers', name='Predições vs Reais',
                  marker=dict(color='green', size=8, opacity=0.6)),
        row=2, col=1
    )
    
    # Linha de referência (y=x)
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                  mode='lines', name='Linha de Referência',
                  line=dict(color='red', width=2, dash='dash')),
        row=2, col=1
    )
    
    # Calcular R²
    r2 = np.corrcoef(actuals, predictions)[0, 1] ** 2
    
    # Atualizar layout
    fig.update_layout(
        title=f'Análise de Predições - {symbol} ({model_type.upper()}) - R² = {r2:.4f}',
        height=800,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Período", row=1, col=1)
    fig.update_yaxes(title_text="Preço ($)", row=1, col=1)
    fig.update_xaxes(title_text="Valores Reais", row=2, col=1)
    fig.update_yaxes(title_text="Predições", row=2, col=1)
    
    return fig

def plot_future_predictions(historical_data, future_predictions, symbol, model_type, days_ahead=30):
    """
    Plota previsões futuras junto com dados históricos
    """
    # Criar datas para o futuro
    last_date = historical_data.index[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot dados históricos
    ax.plot(historical_data.index, historical_data['Close'], 
            'b-', label='Dados Históricos', linewidth=2)
    
    # Plot previsões futuras
    ax.plot(future_dates, future_predictions, 
            'r--', label='Previsões Futuras', linewidth=2)
    
    # Marcar ponto de início das previsões
    ax.axvline(x=last_date, color='g', linestyle=':', alpha=0.7, 
               label='Início das Previsões')
    
    ax.set_title(f'Previsões Futuras - {symbol} ({model_type.upper()})')
    ax.set_xlabel('Data')
    ax.set_ylabel('Preço ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rotacionar labels do eixo x
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Salvar gráfico
    plt.savefig(f'plots/{symbol}_{model_type}_future_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_performance_report(metrics, symbol, model_type):
    """
    Cria um relatório de performance do modelo
    """
    report = f"""
    ========================================
    RELATÓRIO DE PERFORMANCE
    ========================================
    Símbolo: {symbol}
    Modelo: {model_type.upper()}
    Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    MÉTRICAS DE PERFORMANCE:
    - MSE (Mean Squared Error): {metrics['mse']:.4f}
    - RMSE (Root Mean Squared Error): {metrics['rmse']:.4f}
    - MAE (Mean Absolute Error): {metrics['mae']:.4f}
    - R² (Coefficient of Determination): {metrics['r2']:.4f}
    - MAPE (Mean Absolute Percentage Error): {metrics['mape']:.2f}%
    
    INTERPRETAÇÃO:
    - R² próximo a 1 indica boa capacidade preditiva
    - MAPE baixo indica baixo erro percentual
    - RMSE e MAE baixos indicam boa precisão
    """
    
    print(report)
    
    # Salvar relatório
    if not os.path.exists('reports'):
        os.makedirs('reports')
    
    with open(f'reports/{symbol}_{model_type}_performance_report.txt', 'w') as f:
        f.write(report)
    
    return report

def plot_model_comparison(results_dict):
    """
    Compara performance de diferentes modelos
    """
    models = list(results_dict.keys())
    metrics = ['mse', 'rmse', 'mae', 'r2', 'mape']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Comparação de Modelos', fontsize=16)
    
    for i, metric in enumerate(metrics):
        row = i // 3
        col = i % 3
        
        values = [results_dict[model][metric] for model in models]
        
        if metric == 'r2':
            # Para R², valores mais altos são melhores
            colors = ['green' if v > 0.8 else 'orange' if v > 0.6 else 'red' for v in values]
        else:
            # Para outras métricas, valores mais baixos são melhores
            colors = ['green' if v < np.mean(values) else 'orange' if v < np.mean(values) * 1.5 else 'red' for v in values]
        
        bars = axes[row, col].bar(models, values, color=colors, alpha=0.7)
        axes[row, col].set_title(f'{metric.upper()}')
        axes[row, col].set_ylabel(metric.upper())
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[row, col].text(bar.get_x() + bar.get_width()/2., height,
                              f'{value:.4f}', ha='center', va='bottom')
    
    # Remover subplot extra se necessário
    if len(metrics) < 6:
        fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_residuals(actuals, predictions):
    """
    Analisa os resíduos das predições
    """
    residuals = actuals - predictions
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Análise de Resíduos', fontsize=16)
    
    # Plot 1: Resíduos vs Predições
    axes[0, 0].scatter(predictions, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_title('Resíduos vs Predições')
    axes[0, 0].set_xlabel('Predições')
    axes[0, 0].set_ylabel('Resíduos')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Histograma dos resíduos
    axes[0, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Distribuição dos Resíduos')
    axes[0, 1].set_xlabel('Resíduos')
    axes[0, 1].set_ylabel('Frequência')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot dos Resíduos')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Resíduos ao longo do tempo
    axes[1, 1].plot(residuals, 'b-', alpha=0.7)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_title('Resíduos ao Longo do Tempo')
    axes[1, 1].set_xlabel('Período')
    axes[1, 1].set_ylabel('Resíduos')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('plots/residuals_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Estatísticas dos resíduos
    residual_stats = {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'skewness': stats.skew(residuals),
        'kurtosis': stats.kurtosis(residuals)
    }
    
    print("Estatísticas dos Resíduos:")
    for stat, value in residual_stats.items():
        print(f"{stat.capitalize()}: {value:.4f}")
    
    return residual_stats

