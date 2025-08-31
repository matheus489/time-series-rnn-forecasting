"""
Modelo RNN/LSTM para previsão de séries temporais
Implementação usando PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LSTMModel(nn.Module):
    """
    Modelo LSTM para previsão de séries temporais
    """
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Camada LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Camada de saída
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Dropout para regularização
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass do modelo
        
        Args:
            x: Input tensor de shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor de shape (batch_size, output_size)
        """
        # Inicializar estados ocultos
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward pass LSTM
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Pegar apenas o último output da sequência
        lstm_out = lstm_out[:, -1, :]
        
        # Aplicar dropout
        lstm_out = self.dropout(lstm_out)
        
        # Camada de saída
        output = self.fc(lstm_out)
        
        return output

class GRUModel(nn.Module):
    """
    Modelo GRU para previsão de séries temporais (alternativa ao LSTM)
    """
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Camada GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Camada de saída
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Dropout para regularização
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass do modelo
        
        Args:
            x: Input tensor de shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor de shape (batch_size, output_size)
        """
        # Inicializar estado oculto
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward pass GRU
        gru_out, _ = self.gru(x, h0)
        
        # Pegar apenas o último output da sequência
        gru_out = gru_out[:, -1, :]
        
        # Aplicar dropout
        gru_out = self.dropout(gru_out)
        
        # Camada de saída
        output = self.fc(gru_out)
        
        return output

class SimpleRNNModel(nn.Module):
    """
    Modelo RNN simples para comparação
    """
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(SimpleRNNModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Camada RNN
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Camada de saída
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Dropout para regularização
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass do modelo
        
        Args:
            x: Input tensor de shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor de shape (batch_size, output_size)
        """
        # Inicializar estado oculto
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # Forward pass RNN
        rnn_out, _ = self.rnn(x, h0)
        
        # Pegar apenas o último output da sequência
        rnn_out = rnn_out[:, -1, :]
        
        # Aplicar dropout
        rnn_out = self.dropout(rnn_out)
        
        # Camada de saída
        output = self.fc(rnn_out)
        
        return output

class TimeSeriesPredictor:
    """
    Classe wrapper para facilitar o uso dos modelos
    """
    def __init__(self, model_type='lstm', **model_kwargs):
        """
        Inicializa o preditor
        
        Args:
            model_type (str): Tipo de modelo ('lstm', 'gru', 'rnn')
            **model_kwargs: Argumentos para o modelo
        """
        self.model_type = model_type.lower()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Criar modelo
        if self.model_type == 'lstm':
            self.model = LSTMModel(**model_kwargs)
        elif self.model_type == 'gru':
            self.model = GRUModel(**model_kwargs)
        elif self.model_type == 'rnn':
            self.model = SimpleRNNModel(**model_kwargs)
        else:
            raise ValueError(f"Tipo de modelo não suportado: {model_type}")
        
        self.model.to(self.device)
        
    def get_model_summary(self):
        """Retorna um resumo do modelo"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_type': self.model_type,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device)
        }
    
    def save_model(self, filepath):
        """Salva o modelo"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type,
            'model_config': {
                'input_size': self.model.lstm.input_size if hasattr(self.model, 'lstm') else self.model.gru.input_size if hasattr(self.model, 'gru') else self.model.rnn.input_size,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers,
                'output_size': self.model.fc.out_features
            }
        }, filepath)
    
    def load_model(self, filepath):
        """Carrega o modelo"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint

def create_model(model_type='lstm', **kwargs):
    """
    Função factory para criar modelos
    
    Args:
        model_type (str): Tipo de modelo
        **kwargs: Argumentos do modelo
        
    Returns:
        TimeSeriesPredictor: Instância do preditor
    """
    return TimeSeriesPredictor(model_type=model_type, **kwargs)

if __name__ == "__main__":
    # Teste dos modelos
    print("Testando modelos...")
    
    # Criar dados de teste
    batch_size = 32
    sequence_length = 60
    input_size = 1
    
    x = torch.randn(batch_size, sequence_length, input_size)
    
    # Testar diferentes tipos de modelo
    model_types = ['lstm', 'gru', 'rnn']
    
    for model_type in model_types:
        print(f"\nTestando modelo {model_type.upper()}:")
        
        model = create_model(
            model_type=model_type,
            input_size=input_size,
            hidden_size=50,
            num_layers=2,
            output_size=1
        )
        
        # Forward pass
        with torch.no_grad():
            output = model.model(x)
        
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Model summary: {model.get_model_summary()}")

