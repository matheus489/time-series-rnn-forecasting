"""
Script de treinamento para modelos RNN de séries temporais
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar módulos locais
from data_loader import TimeSeriesDataLoader
from model import create_model
from utils import plot_training_history, plot_predictions

class TimeSeriesTrainer:
    """
    Classe para treinamento de modelos de séries temporais
    """
    def __init__(self, model_type='lstm', **model_kwargs):
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Usando dispositivo: {self.device}")
        
        # Criar modelo
        self.predictor = create_model(model_type=model_type, **model_kwargs)
        self.model = self.predictor.model
        
        # Histórico de treinamento
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
    def prepare_data_loaders(self, X_train, y_train, X_val, y_val, batch_size=32):
        """
        Prepara DataLoaders para treinamento
        """
        # Converter para tensores
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # Criar datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # Criar dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """
        Treina uma época
        """
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = self.model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Guardar predições para métricas
            outputs_np = outputs.squeeze().cpu().detach().numpy()
            if outputs_np.ndim == 0:  # Se for escalar, converter para array
                outputs_np = np.array([outputs_np])
            
            batch_y_np = batch_y.cpu().numpy()
            if batch_y_np.ndim == 0:  # Se for escalar, converter para array
                batch_y_np = np.array([batch_y_np])
            
            predictions.extend(outputs_np)
            targets.extend(batch_y_np)
        
        avg_loss = total_loss / len(train_loader)
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        return avg_loss, mse, mae, r2
    
    def validate_epoch(self, val_loader, criterion):
        """
        Valida uma época
        """
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                
                total_loss += loss.item()
                
                # Garantir que outputs seja um tensor 1D
                outputs_np = outputs.squeeze().cpu().numpy()
                if outputs_np.ndim == 0:  # Se for escalar, converter para array
                    outputs_np = np.array([outputs_np])
                
                batch_y_np = batch_y.cpu().numpy()
                if batch_y_np.ndim == 0:  # Se for escalar, converter para array
                    batch_y_np = np.array([batch_y_np])
                
                predictions.extend(outputs_np)
                targets.extend(batch_y_np)
        
        avg_loss = total_loss / len(val_loader)
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)
        
        return avg_loss, mse, mae, r2
    
    def train(self, train_loader, val_loader, epochs=100, learning_rate=0.001, 
              patience=10, save_best=True, model_dir='models'):
        """
        Treina o modelo
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Iniciando treinamento do modelo {self.model_type.upper()}")
        print(f"Épocas: {epochs}, Learning Rate: {learning_rate}")
        print(f"{'='*60}")
        
        for epoch in range(epochs):
            # Treinar
            train_loss, train_mse, train_mae, train_r2 = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validar
            val_loss, val_mse, val_mae, val_r2 = self.validate_epoch(val_loader, criterion)
            
            # Atualizar learning rate
            scheduler.step(val_loss)
            
            # Guardar métricas
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_metrics.append({'mse': train_mse, 'mae': train_mae, 'r2': train_r2})
            self.val_metrics.append({'mse': val_mse, 'mae': val_mae, 'r2': val_r2})
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Salvar melhor modelo
                if save_best:
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir)
                    model_path = os.path.join(model_dir, f'best_{self.model_type}_model.pth')
                    self.predictor.save_model(model_path)
            else:
                patience_counter += 1
            
            # Print progresso
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Época {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"Train R²: {train_r2:.4f} | "
                      f"Val R²: {val_r2:.4f}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping na época {epoch+1}")
                break
        
        print(f"\nTreinamento concluído!")
        print(f"Melhor validação loss: {best_val_loss:.6f}")
        
        return self.train_losses, self.val_losses, self.train_metrics, self.val_metrics
    
    def evaluate(self, X_test, y_test, scaler):
        """
        Avalia o modelo no conjunto de teste
        """
        self.model.eval()
        
        # Converter para tensor
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        # Fazer predições
        with torch.no_grad():
            predictions = self.model(X_test_tensor).squeeze().cpu().numpy()
        
        # Desnormalizar predições
        predictions_denorm = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        y_test_denorm = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Calcular métricas
        mse = mean_squared_error(y_test_denorm, predictions_denorm)
        mae = mean_absolute_error(y_test_denorm, predictions_denorm)
        r2 = r2_score(y_test_denorm, predictions_denorm)
        
        # Calcular erro percentual médio
        mape = np.mean(np.abs((y_test_denorm - predictions_denorm) / y_test_denorm)) * 100
        
        metrics = {
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'rmse': np.sqrt(mse)
        }
        
        print(f"\nMétricas de Teste:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        return predictions_denorm, y_test_denorm, metrics
    
    def predict_future(self, last_sequence, scaler, steps=30):
        """
        Faz previsões para o futuro
        """
        self.model.eval()
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        with torch.no_grad():
            for _ in range(steps):
                # Preparar input
                input_tensor = torch.FloatTensor(current_sequence.reshape(1, -1, 1)).to(self.device)
                
                # Fazer predição
                prediction = self.model(input_tensor).squeeze().cpu().numpy()
                predictions.append(prediction)
                
                # Atualizar sequência (remover primeiro elemento e adicionar predição)
                current_sequence = np.roll(current_sequence, -1)
                current_sequence[-1] = prediction
        
        # Desnormalizar predições
        predictions = np.array(predictions)
        predictions_denorm = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        return predictions_denorm

def main():
    """
    Função principal para demonstrar o treinamento
    """
    # Configurações
    symbol = 'AAPL'
    model_type = 'lstm'
    epochs = 100
    batch_size = 32
    learning_rate = 0.001
    
    print(f"Treinando modelo {model_type.upper()} para {symbol}")
    print(f"{'='*60}")
    
    # Carregar dados
    loader = TimeSeriesDataLoader()
    
    try:
        # Tentar carregar dados processados
        processed_data = loader.load_processed_data(f'{symbol}_processed.pkl')
        print(f"Dados carregados de {symbol}_processed.pkl")
    except FileNotFoundError:
        print(f"Dados não encontrados. Baixando e processando {symbol}...")
        
        # Baixar e processar dados
        data = loader.download_stock_data(symbol, period='2y')
        X, y = loader.prepare_sequences(data, sequence_length=60)
        X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(X, y)
        
        # Salvar dados processados
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
        loader.save_processed_data(processed_data, f'{symbol}_processed.pkl')
    
    # Extrair dados
    X_train = processed_data['X_train']
    X_val = processed_data['X_val']
    X_test = processed_data['X_test']
    y_train = processed_data['y_train']
    y_val = processed_data['y_val']
    y_test = processed_data['y_test']
    scaler = processed_data['scaler']
    
    # Criar trainer
    trainer = TimeSeriesTrainer(
        model_type=model_type,
        input_size=1,
        hidden_size=50,
        num_layers=2,
        output_size=1,
        dropout=0.2
    )
    
    # Preparar dataloaders
    train_loader, val_loader = trainer.prepare_data_loaders(
        X_train, y_train, X_val, y_val, batch_size=batch_size
    )
    
    # Treinar modelo
    train_losses, val_losses, train_metrics, val_metrics = trainer.train(
        train_loader, val_loader, 
        epochs=epochs, 
        learning_rate=learning_rate,
        patience=15
    )
    
    # Avaliar modelo
    predictions, actuals, test_metrics = trainer.evaluate(X_test, y_test, scaler)
    
    # Salvar modelo final
    model_path = f'models/{symbol}_{model_type}_final_model.pth'
    trainer.predictor.save_model(model_path)
    print(f"Modelo final salvo em: {model_path}")
    
    # Plotar resultados
    plot_training_history(train_losses, val_losses, train_metrics, val_metrics, symbol, model_type)
    plot_predictions(actuals, predictions, symbol, model_type)
    
    # Fazer previsões futuras
    last_sequence = X_test[-1]  # Última sequência do teste
    future_predictions = trainer.predict_future(last_sequence, scaler, steps=30)
    
    print(f"\nPrevisões para os próximos 30 dias:")
    for i, pred in enumerate(future_predictions[:10]):  # Mostrar apenas os primeiros 10
        print(f"Dia {i+1}: ${pred:.2f}")
    
    return trainer, test_metrics

if __name__ == "__main__":
    main()

