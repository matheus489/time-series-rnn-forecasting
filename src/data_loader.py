"""
Carregador de dados para séries temporais
Baixa dados de ações via múltiplas APIs e pré-processa para treinamento
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from datetime import datetime, timedelta
import pickle
import requests
import time
import json

class TimeSeriesDataLoader:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Criar diretório se não existir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Carregar API keys se disponíveis
        self.api_keys = self._load_api_keys()
    
    def _load_api_keys(self):
        """Carrega API keys do arquivo de configuração"""
        config_file = 'api_config.json'
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Erro ao carregar configuração: {e}")
        
        return {}
    
    def download_stock_data(self, symbol='AAPL', period='2y', max_retries=3):
        """
        Baixa dados de ações de múltiplas fontes com fallback
        
        Args:
            symbol (str): Símbolo da ação (ex: 'AAPL', 'GOOGL', 'MSFT')
            period (str): Período de dados ('1y', '2y', '5y', etc.)
            max_retries (int): Número máximo de tentativas
        """
        print(f"Baixando dados para {symbol}...")
        
        # Tentar diferentes fontes de dados
        data_sources = [
            ('Yahoo Finance', self._download_from_yahoo),
            ('Alpha Vantage', self._download_from_alphavantage),
            ('Polygon.io', self._download_from_polygon),
            ('Finnhub', self._download_from_finnhub),
            ('IEX Cloud', self._download_from_iex),
            ('Quandl', self._download_from_quandl)
        ]
        
        for source_name, download_func in data_sources:
            try:
                print(f"Tentando {source_name}...")
                data = download_func(symbol, period)
                if data is not None and not data.empty and len(data) > 10:
                    print(f"✅ Dados baixados com sucesso via {source_name}")
                    
                    # Salvar dados brutos
                    file_path = os.path.join(self.data_dir, f'{symbol}_raw_data.csv')
                    data.to_csv(file_path)
                    print(f"Dados salvos em: {file_path}")
                    
                    return data
                else:
                    print(f"⚠️ {source_name}: dados insuficientes")
                    
            except Exception as e:
                print(f"❌ {source_name}: {str(e)}")
                continue
        
        # Se chegou aqui, todas as fontes falharam
        print("🔄 Todas as fontes falharam. Usando dados sintéticos...")
        return self.generate_synthetic_data(symbol, period)
    
    def _download_from_yahoo(self, symbol, period):
        """Baixa dados do Yahoo Finance"""
        # Lista de símbolos alternativos para tentar
        alternative_symbols = {
            'AAPL': ['AAPL', 'AAPL.US', 'AAPL.O', 'AAPL.SA'],
            'GOOGL': ['GOOGL', 'GOOG', 'GOOGL.US', 'GOOGL.SA'],
            'MSFT': ['MSFT', 'MSFT.US', 'MSFT.O', 'MSFT.SA'],
            'TSLA': ['TSLA', 'TSLA.US', 'TSLA.O', 'TSLA.SA'],
            'AMZN': ['AMZN', 'AMZN.US', 'AMZN.O', 'AMZN.SA']
        }
        
        symbols_to_try = alternative_symbols.get(symbol, [symbol])
        periods_to_try = [period, '1y', '6mo', '3mo']
        
        for current_symbol in symbols_to_try:
            for current_period in periods_to_try:
                try:
                    ticker = yf.Ticker(current_symbol)
                    data = ticker.history(period=current_period)
                    
                    if not data.empty and len(data) > 10:
                        return data
                        
                except Exception:
                    continue
        
        return None
    
    def _download_from_alphavantage(self, symbol, period):
        """Baixa dados do Alpha Vantage (gratuito com limite de requisições)"""
        try:
            # Usar API key configurada ou demo
            api_key = self.api_keys.get('alpha_vantage', 'demo')
            
            # Mapear período para intervalo
            interval_map = {
                '1mo': 'daily',
                '3mo': 'daily', 
                '6mo': 'daily',
                '1y': 'daily',
                '2y': 'daily',
                '5y': 'daily'
            }
            
            interval = interval_map.get(period, 'daily')
            
            # URL da API
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': api_key,
                'outputsize': 'full'
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'Time Series (Daily)' in data:
                # Converter para DataFrame
                df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
                df.index = pd.to_datetime(df.index)
                
                # Renomear colunas
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                df = df.astype(float)
                
                # Filtrar por período
                days = self._period_to_days(period)
                if days:
                    start_date = datetime.now() - timedelta(days=days)
                    df = df[df.index >= start_date]
                
                return df
                
        except Exception as e:
            print(f"Alpha Vantage error: {e}")
        
        return None
    
    def _download_from_polygon(self, symbol, period):
        """Baixa dados do Polygon.io (gratuito com limite)"""
        try:
            # Usar API key configurada ou demo
            api_key = self.api_keys.get('polygon', 'demo')
            
            # Calcular datas
            end_date = datetime.now()
            days = self._period_to_days(period)
            start_date = end_date - timedelta(days=days)
            
            # URL da API
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {'apiKey': api_key}
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get('status') == 'OK' and 'results' in data:
                # Converter para DataFrame
                results = data['results']
                df = pd.DataFrame(results)
                
                # Converter timestamp para datetime
                df['t'] = pd.to_datetime(df['t'], unit='ms')
                df.set_index('t', inplace=True)
                
                # Renomear colunas
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'VWAP', 'Transactions']
                df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                
                return df
                
        except Exception as e:
            print(f"Polygon.io error: {e}")
        
        return None
    
    def _download_from_finnhub(self, symbol, period):
        """Baixa dados do Finnhub (gratuito com limite)"""
        try:
            # Usar API key configurada ou demo
            api_key = self.api_keys.get('finnhub', 'demo')
            
            # Calcular datas
            end_date = int(datetime.now().timestamp())
            days = self._period_to_days(period)
            start_date = int((datetime.now() - timedelta(days=days)).timestamp())
            
            # URL da API
            url = f"https://finnhub.io/api/v1/stock/candle"
            params = {
                'symbol': symbol,
                'resolution': 'D',
                'from': start_date,
                'to': end_date,
                'token': api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get('s') == 'ok' and 't' in data:
                # Converter para DataFrame
                df = pd.DataFrame({
                    'Open': data['o'],
                    'High': data['h'],
                    'Low': data['l'],
                    'Close': data['c'],
                    'Volume': data['v']
                })
                
                # Converter timestamp para datetime
                df.index = pd.to_datetime(data['t'], unit='s')
                
                return df
                
        except Exception as e:
            print(f"Finnhub error: {e}")
        
        return None
    
    def _download_from_iex(self, symbol, period):
        """Baixa dados do IEX Cloud (gratuito com limite)"""
        try:
            # Usar API key configurada ou demo
            api_key = self.api_keys.get('iex', 'demo')
            
            # Calcular datas
            end_date = datetime.now()
            days = self._period_to_days(period)
            start_date = end_date - timedelta(days=days)
            
            # URL da API
            url = f"https://cloud.iexapis.com/stable/stock/{symbol}/chart/{days}d"
            params = {'token': api_key}
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                # Converter para DataFrame
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # Renomear colunas
                df = df[['open', 'high', 'low', 'close', 'volume']]
                df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                
                return df
                
        except Exception as e:
            print(f"IEX Cloud error: {e}")
        
        return None
    
    def _download_from_quandl(self, symbol, period):
        """Baixa dados do Quandl (gratuito com limite)"""
        try:
            # Usar API key configurada ou demo
            api_key = self.api_keys.get('quandl', 'demo')
            
            # Calcular datas
            end_date = datetime.now()
            days = self._period_to_days(period)
            start_date = end_date - timedelta(days=days)
            
            # URL da API
            url = f"https://www.quandl.com/api/v3/datasets/WIKI/{symbol}/data.json"
            params = {
                'api_key': api_key,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'dataset_data' in data and 'data' in data['dataset_data']:
                # Converter para DataFrame
                columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                df = pd.DataFrame(data['dataset_data']['data'], columns=columns)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                
                return df
                
        except Exception as e:
            print(f"Quandl error: {e}")
        
        return None
    
    def _period_to_days(self, period):
        """Converte período para número de dias"""
        period_map = {
            '1mo': 30,
            '3mo': 90,
            '6mo': 180,
            '1y': 365,
            '2y': 730,
            '5y': 1825
        }
        return period_map.get(period, 365)

    def prepare_sequences(self, data, target_column='Close', sequence_length=60):
        """
        Prepara sequências para treinamento do modelo RNN
        
        Args:
            data (pd.DataFrame): Dados da série temporal
            target_column (str): Coluna alvo para previsão
            sequence_length (int): Comprimento da sequência de entrada
            
        Returns:
            tuple: (X, y) arrays para treinamento
        """
        # Usar apenas a coluna alvo
        target_data = data[target_column].values.reshape(-1, 1)
        
        # Normalizar dados
        scaled_data = self.scaler.fit_transform(target_data)
        
        # Criar sequências
        X, y = [], []
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape para (samples, sequence_length, features)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def split_data(self, X, y, train_split=0.8, val_split=0.1):
        """
        Divide os dados em treino, validação e teste
        
        Args:
            X (np.array): Features
            y (np.array): Targets
            train_split (float): Proporção para treino
            val_split (float): Proporção para validação
            
        Returns:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        total_samples = len(X)
        train_size = int(total_samples * train_split)
        val_size = int(total_samples * val_split)
        
        # Dividir dados
        X_train = X[:train_size]
        y_train = y[:train_size]
        
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        print(f"Divisão dos dados:")
        print(f"Treino: {len(X_train)} amostras")
        print(f"Validação: {len(X_val)} amostras")
        print(f"Teste: {len(X_test)} amostras")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(self, data_dict, filename):
        """Salva dados processados"""
        file_path = os.path.join(self.data_dir, filename)
        with open(file_path, 'wb') as f:
            pickle.dump(data_dict, f)
        print(f"Dados processados salvos em: {file_path}")
    
    def load_processed_data(self, filename):
        """Carrega dados processados"""
        file_path = os.path.join(self.data_dir, filename)
        with open(file_path, 'rb') as f:
            data_dict = pickle.load(f)
        return data_dict
    
    def get_stock_info(self, symbol):
        """Obtém informações básicas da ação"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A')
            }
        except:
            return {
                'name': symbol,
                'sector': 'N/A',
                'industry': 'N/A',
                'market_cap': 'N/A'
            }

    def generate_synthetic_data(self, symbol='SYNTHETIC', period='1y'):
        """
        Gera dados sintéticos para demonstração quando o download falha
        
        Args:
            symbol (str): Nome do símbolo sintético
            period (str): Período simulado
            
        Returns:
            pd.DataFrame: Dados sintéticos
        """
        print(f"Gerando dados sintéticos para demonstração...")
        
        # Calcular número de dias baseado no período
        period_days = {
            '1y': 365,
            '2y': 730,
            '6mo': 180,
            '3mo': 90
        }
        
        days = period_days.get(period, 365)
        
        # Gerar datas
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Gerar preços sintéticos com tendência e volatilidade realistas
        np.random.seed(42)  # Para reprodutibilidade
        
        # Preço inicial
        initial_price = 150.0
        
        # Gerar retornos logarítmicos com drift e volatilidade
        daily_returns = np.random.normal(0.0005, 0.02, len(dates))  # 0.05% drift, 2% volatilidade
        
        # Adicionar tendência de longo prazo
        trend = np.linspace(0, 0.1, len(dates))  # 10% de crescimento no período
        daily_returns += trend / len(dates)
        
        # Calcular preços
        prices = [initial_price]
        for ret in daily_returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Criar DataFrame
        data = pd.DataFrame({
            'Open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'High': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
        
        # Garantir que High >= Low e High >= Close >= Low
        data['High'] = data[['Open', 'Close', 'High']].max(axis=1)
        data['Low'] = data[['Open', 'Close', 'Low']].min(axis=1)
        
        # Salvar dados sintéticos
        file_path = os.path.join(self.data_dir, f'{symbol}_synthetic_data.csv')
        data.to_csv(file_path)
        print(f"Dados sintéticos salvos em: {file_path}")
        
        return data

def main():
    """Função principal para demonstrar o uso"""
    # Inicializar carregador
    loader = TimeSeriesDataLoader()
    
    # Lista de ações populares para teste
    symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
    
    for symbol in symbols:
        try:
            print(f"\n{'='*50}")
            print(f"Processando {symbol}")
            print(f"{'='*50}")
            
            # Baixar dados
            data = loader.download_stock_data(symbol, period='2y')
            
            # Obter informações da ação
            info = loader.get_stock_info(symbol)
            print(f"Nome: {info['name']}")
            print(f"Setor: {info['sector']}")
            print(f"Indústria: {info['industry']}")
            
            # Preparar sequências
            X, y = loader.prepare_sequences(data, sequence_length=60)
            
            # Dividir dados
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
                'symbol': symbol,
                'info': info
            }
            
            loader.save_processed_data(processed_data, f'{symbol}_processed.pkl')
            
        except Exception as e:
            print(f"Erro ao processar {symbol}: {str(e)}")
            continue

if __name__ == "__main__":
    main()

