"""
Aplicação Streamlit para visualização interativa de previsões de séries temporais
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import pickle
import os
import sys

# Adicionar o diretório src ao path para importar módulos locais
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import TimeSeriesDataLoader
from model import create_model
from trainer import TimeSeriesTrainer
from utils import plot_interactive_predictions, create_performance_report

# Configuração da página
st.set_page_config(
    page_title="Previsão de Séries Temporais com RNN",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("📈 Previsão de Séries Temporais com RNN")
st.markdown("---")

# Sidebar para configurações
st.sidebar.header("⚙️ Configurações")

# Seleção de ação
available_symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NVDA', 'NFLX']
selected_symbol = st.sidebar.selectbox(
    "Selecione uma ação:",
    available_symbols,
    index=0
)

# Seleção do modelo
model_types = ['lstm', 'gru', 'rnn']
selected_model = st.sidebar.selectbox(
    "Tipo de modelo:",
    model_types,
    index=0
)

# Parâmetros do modelo
st.sidebar.subheader("🔧 Parâmetros do Modelo")
hidden_size = st.sidebar.slider("Tamanho da camada oculta:", 20, 100, 50)
num_layers = st.sidebar.slider("Número de camadas:", 1, 4, 2)
dropout = st.sidebar.slider("Dropout:", 0.0, 0.5, 0.2, 0.1)

# Parâmetros de treinamento
st.sidebar.subheader("🎯 Parâmetros de Treinamento")
epochs = st.sidebar.slider("Épocas:", 10, 200, 50)
batch_size = st.sidebar.slider("Batch size:", 16, 64, 32)
learning_rate = st.sidebar.selectbox(
    "Learning rate:",
    [0.0001, 0.0005, 0.001, 0.005, 0.01],
    index=2
)

# Período de dados
period = st.sidebar.selectbox(
    "Período de dados:",
    ['1y', '2y', '5y'],
    index=1
)

# Botão para treinar modelo
train_button = st.sidebar.button("🚀 Treinar Modelo", type="primary")

# Função para carregar ou baixar dados
def load_stock_data(symbol, period):
    """Carrega dados da ação"""
    try:
        # Tentar carregar dados processados
        loader = TimeSeriesDataLoader()
        processed_data = loader.load_processed_data(f'{symbol}_processed.pkl')
        return processed_data, True
    except FileNotFoundError:
        # Baixar dados se não existirem
        loader = TimeSeriesDataLoader()
        
        # Tentar baixar dados reais, se falhar usar sintéticos
        try:
            data = loader.download_stock_data(symbol, period)
            st.info(f"✅ Dados reais baixados para {symbol}")
        except Exception as e:
            st.warning(f"⚠️ Erro ao baixar dados reais: {str(e)}")
            st.info("🔄 Usando dados sintéticos para demonstração...")
            data = loader.generate_synthetic_data(symbol, period)
        
        X, y = loader.prepare_sequences(data, sequence_length=60)
        X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X, y)
        
        processed_data = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'scaler': loader.scaler,
            'raw_data': data,
            'symbol': symbol
        }
        return processed_data, False

# Função para treinar modelo
def train_model(processed_data, model_params, train_params):
    """Treina o modelo com os parâmetros especificados"""
    # Criar trainer
    trainer = TimeSeriesTrainer(
        model_type=model_params['model_type'],
        input_size=1,
        hidden_size=model_params['hidden_size'],
        num_layers=model_params['num_layers'],
        output_size=1,
        dropout=model_params['dropout']
    )
    
    # Preparar dataloaders
    train_loader, val_loader = trainer.prepare_data_loaders(
        processed_data['X_train'], processed_data['y_train'],
        processed_data['X_val'], processed_data['y_val'],
        batch_size=train_params['batch_size']
    )
    
    # Treinar modelo
    train_losses, val_losses, train_metrics, val_metrics = trainer.train(
        train_loader, val_loader,
        epochs=train_params['epochs'],
        learning_rate=train_params['learning_rate'],
        patience=15
    )
    
    # Avaliar modelo
    predictions, actuals, test_metrics = trainer.evaluate(
        processed_data['X_test'], processed_data['y_test'], processed_data['scaler']
    )
    
    return trainer, predictions, actuals, test_metrics, train_losses, val_losses

# Função para fazer previsões futuras
def make_future_predictions(trainer, processed_data, days_ahead=30):
    """Faz previsões para o futuro"""
    last_sequence = processed_data['X_test'][-1]
    future_predictions = trainer.predict_future(last_sequence, processed_data['scaler'], days_ahead)
    
    # Criar datas futuras
    last_date = processed_data['raw_data'].index[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(days_ahead)]
    
    return future_predictions, future_dates

# Interface principal
if train_button:
    st.header(f"🎯 Treinando modelo {selected_model.upper()} para {selected_symbol}")
    
    # Mostrar informações da ação
    with st.spinner("Carregando dados..."):
        processed_data, is_cached = load_stock_data(selected_symbol, period)
        
        if is_cached:
            st.success(f"✅ Dados carregados do cache para {selected_symbol}")
        else:
            st.success(f"✅ Dados baixados e processados para {selected_symbol}")
    
    # Informações da ação
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Período de dados", f"{len(processed_data['raw_data'])} dias")
    with col2:
        st.metric("Dados de treino", f"{len(processed_data['X_train'])} amostras")
    with col3:
        st.metric("Dados de teste", f"{len(processed_data['X_test'])} amostras")
    
    # Parâmetros do modelo
    model_params = {
        'model_type': selected_model,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'dropout': dropout
    }
    
    train_params = {
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    
    # Treinar modelo
    with st.spinner("Treinando modelo..."):
        trainer, predictions, actuals, test_metrics, train_losses, val_losses = train_model(
            processed_data, model_params, train_params
        )
    
    st.success("✅ Modelo treinado com sucesso!")
    
    # Métricas de performance
    st.subheader("📊 Métricas de Performance")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("MSE", f"{test_metrics['mse']:.4f}")
    with col2:
        st.metric("RMSE", f"{test_metrics['rmse']:.4f}")
    with col3:
        st.metric("MAE", f"{test_metrics['mae']:.4f}")
    with col4:
        st.metric("R²", f"{test_metrics['r2']:.4f}")
    with col5:
        st.metric("MAPE", f"{test_metrics['mape']:.2f}%")
    
    # Gráficos
    st.subheader("📈 Visualizações")
    
    # Tabs para diferentes visualizações
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Predições", "📈 Histórico de Treinamento", "🔮 Previsões Futuras", "📋 Relatório"])
    
    with tab1:
        # Gráfico interativo de predições
        fig = plot_interactive_predictions(actuals, predictions, selected_symbol, selected_model)
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot adicional
        fig_scatter = px.scatter(
            x=actuals, y=predictions,
            title=f"Scatter Plot: Valores Reais vs Predições - {selected_symbol}",
            labels={'x': 'Valores Reais', 'y': 'Predições'}
        )
        fig_scatter.add_trace(go.Scatter(
            x=[actuals.min(), actuals.max()],
            y=[actuals.min(), actuals.max()],
            mode='lines',
            name='Linha de Referência',
            line=dict(color='red', dash='dash')
        ))
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab2:
        # Histórico de treinamento
        fig_history = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Loss', 'MSE', 'MAE', 'R²'),
            vertical_spacing=0.1
        )
        
        epochs_range = list(range(1, len(train_losses) + 1))
        
        # Loss
        fig_history.add_trace(
            go.Scatter(x=epochs_range, y=train_losses, name='Treino', line=dict(color='blue')),
            row=1, col=1
        )
        fig_history.add_trace(
            go.Scatter(x=epochs_range, y=val_losses, name='Validação', line=dict(color='red')),
            row=1, col=1
        )
        
        # MSE
        train_mse = [m['mse'] for m in trainer.train_metrics]
        val_mse = [m['mse'] for m in trainer.val_metrics]
        fig_history.add_trace(
            go.Scatter(x=epochs_range, y=train_mse, name='Treino MSE', line=dict(color='blue'), showlegend=False),
            row=1, col=2
        )
        fig_history.add_trace(
            go.Scatter(x=epochs_range, y=val_mse, name='Validação MSE', line=dict(color='red'), showlegend=False),
            row=1, col=2
        )
        
        # MAE
        train_mae = [m['mae'] for m in trainer.train_metrics]
        val_mae = [m['mae'] for m in trainer.val_metrics]
        fig_history.add_trace(
            go.Scatter(x=epochs_range, y=train_mae, name='Treino MAE', line=dict(color='blue'), showlegend=False),
            row=2, col=1
        )
        fig_history.add_trace(
            go.Scatter(x=epochs_range, y=val_mae, name='Validação MAE', line=dict(color='red'), showlegend=False),
            row=2, col=1
        )
        
        # R²
        train_r2 = [m['r2'] for m in trainer.train_metrics]
        val_r2 = [m['r2'] for m in trainer.val_metrics]
        fig_history.add_trace(
            go.Scatter(x=epochs_range, y=train_r2, name='Treino R²', line=dict(color='blue'), showlegend=False),
            row=2, col=2
        )
        fig_history.add_trace(
            go.Scatter(x=epochs_range, y=val_r2, name='Validação R²', line=dict(color='red'), showlegend=False),
            row=2, col=2
        )
        
        fig_history.update_layout(height=600, title_text="Histórico de Treinamento")
        st.plotly_chart(fig_history, use_container_width=True)
    
    with tab3:
        # Previsões futuras
        days_ahead = st.slider("Dias para prever:", 7, 90, 30)
        
        if st.button("🔮 Fazer Previsões Futuras"):
            with st.spinner("Fazendo previsões futuras..."):
                future_predictions, future_dates = make_future_predictions(
                    trainer, processed_data, days_ahead
                )
            
            # Gráfico de previsões futuras
            fig_future = go.Figure()
            
            # Dados históricos
            fig_future.add_trace(go.Scatter(
                x=processed_data['raw_data'].index,
                y=processed_data['raw_data']['Close'],
                mode='lines',
                name='Dados Históricos',
                line=dict(color='blue', width=2)
            ))
            
            # Previsões futuras
            fig_future.add_trace(go.Scatter(
                x=future_dates,
                y=future_predictions,
                mode='lines',
                name='Previsões Futuras',
                line=dict(color='red', width=2, dash='dash')
            ))
            
            # Linha de separação
            fig_future.add_vline(
                x=processed_data['raw_data'].index[-1],
                line_dash="dash",
                line_color="green",
                annotation_text="Início das Previsões"
            )
            
            fig_future.update_layout(
                title=f"Previsões Futuras - {selected_symbol} ({selected_model.upper()})",
                xaxis_title="Data",
                yaxis_title="Preço ($)",
                height=500
            )
            
            st.plotly_chart(fig_future, use_container_width=True)
            
            # Tabela com previsões
            st.subheader("📋 Previsões Detalhadas")
            predictions_df = pd.DataFrame({
                'Data': future_dates,
                'Preço Previsto ($)': [f"${pred:.2f}" for pred in future_predictions],
                'Variação (%)': [f"{((pred - future_predictions[0]) / future_predictions[0] * 100):.2f}%" 
                               for pred in future_predictions]
            })
            st.dataframe(predictions_df, use_container_width=True)
    
    with tab4:
        # Relatório de performance
        report = create_performance_report(test_metrics, selected_symbol, selected_model)
        st.text(report)
        
        # Download do relatório
        st.download_button(
            label="📥 Baixar Relatório",
            data=report,
            file_name=f"{selected_symbol}_{selected_model}_report.txt",
            mime="text/plain"
        )

# Página inicial quando não há treinamento
else:
    st.markdown("""
    ## 🎯 Bem-vindo ao Sistema de Previsão de Séries Temporais!
    
    Este sistema utiliza **Redes Neurais Recorrentes (RNN)** para prever preços de ações.
    
    ### 🚀 Como usar:
    1. **Selecione uma ação** na barra lateral
    2. **Escolha o tipo de modelo** (LSTM, GRU ou RNN)
    3. **Ajuste os parâmetros** conforme necessário
    4. **Clique em 'Treinar Modelo'** para iniciar o processo
    
    ### 📊 Funcionalidades:
    - **Carregamento automático de dados** via Yahoo Finance
    - **Treinamento de modelos RNN** com diferentes arquiteturas
    - **Visualizações interativas** das predições
    - **Análise de performance** com múltiplas métricas
    - **Previsões futuras** para planejamento
    
    ### 🔧 Modelos Disponíveis:
    - **LSTM**: Long Short-Term Memory - ideal para séries temporais longas
    - **GRU**: Gated Recurrent Unit - mais eficiente que LSTM
    - **RNN**: Rede Neural Recorrente simples - para comparação
    
    ### 📈 Métricas de Avaliação:
    - **MSE**: Erro quadrático médio
    - **RMSE**: Raiz do erro quadrático médio
    - **MAE**: Erro absoluto médio
    - **R²**: Coeficiente de determinação
    - **MAPE**: Erro percentual absoluto médio
    """)
    
    # Exemplo de dados
    st.subheader("📊 Exemplo de Dados Disponíveis")
    
    # Carregar dados de exemplo
    try:
        example_data, _ = load_stock_data('AAPL', '1y')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dados de exemplo (AAPL):**")
            st.dataframe(example_data['raw_data'].tail(), use_container_width=True)
        
        with col2:
            st.write("**Estatísticas:**")
            stats_df = example_data['raw_data']['Close'].describe()
            st.dataframe(stats_df, use_container_width=True)
    
    except Exception as e:
        st.info("Execute o treinamento para ver dados de exemplo.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Desenvolvido com ❤️ usando Streamlit, PyTorch e Yahoo Finance</p>
        <p>📈 Sistema de Previsão de Séries Temporais com RNN</p>
    </div>
    """,
    unsafe_allow_html=True
)

