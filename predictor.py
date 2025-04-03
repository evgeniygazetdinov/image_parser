import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import os
import joblib
from datetime import datetime
import json
import shutil

class RealEstatePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.model_dir = 'models'
        self.best_models_dir = os.path.join(self.model_dir, 'best_models')
        self.metrics_file = os.path.join(self.model_dir, 'model_metrics.json')
        self.best_mae = float('inf')
        
        # Создаем необходимые директории
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.best_models_dir, exist_ok=True)
        
        # Загружаем историю метрик, если она существует
        self.load_metrics_history()
    
    def load_metrics_history(self):
        """Загрузка истории метрик моделей"""
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r') as f:
                self.metrics_history = json.load(f)
        else:
            self.metrics_history = {}
    
    def save_metrics_history(self):
        """Сохранение истории метрик моделей"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=4)
    
    def preprocess_data(self, df):
        # Копируем датафрейм
        df_processed = df.copy()
        
        # Категориальные признаки
        categorical_features = ['Apartment type', 'Metro station', 'Region', 'Renovation']
        
        # Кодируем категориальные признаки
        for feature in categorical_features:
            if feature in df_processed.columns:
                le = LabelEncoder()
                df_processed[feature] = le.fit_transform(df_processed[feature].astype(str))
                self.label_encoders[feature] = le
        
        return df_processed
    
    def create_model(self, input_dim):
        model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)  # Выходной слой для предсказания цены
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mean_squared_error',
                     metrics=['mae'])
        
        self.model = model
        return model
    
    def prepare_data(self, df):
        # Предобработка данных
        df_processed = self.preprocess_data(df)
        
        # Разделяем признаки и целевую переменную
        X = df_processed.drop('Price', axis=1)
        y = df_processed['Price']
        
        # Нормализуем числовые признаки
        X_scaled = self.scaler.fit_transform(X)
        
        # Разделяем на тренировочную и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test, X.shape[1]
    
    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2):
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        # Сохраняем метрики после обучения
        val_loss = min(history.history['val_loss'])
        val_mae = min(history.history['val_mae'])
        
        metrics = {
            'val_loss': float(val_loss),
            'mae': float(val_mae),
            'epochs': epochs,
            'batch_size': batch_size,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Сохраняем модель с метриками
        self.save_model(metrics=metrics)
        
        return history
    
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = self.preprocess_data(X)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def plot_training_history(self, history):
        plt.figure(figsize=(12, 4))
        
        # График ошибки
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Ошибка модели')
        plt.ylabel('Ошибка')
        plt.xlabel('Эпоха')
        plt.legend(['Тренировочная', 'Валидационная'])
        
        # График MAE
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'])
        plt.plot(history.history['val_mae'])
        plt.title('Средняя абсолютная ошибка')
        plt.ylabel('MAE')
        plt.xlabel('Эпоха')
        plt.legend(['Тренировочная', 'Валидационная'])
        
        plt.tight_layout()
        plt.show()

    def save_model(self, model_name=None, metrics=None):
        """Сохранение модели и всех preprocessors с метриками качества"""
        if model_name is None:
            model_name = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Создаем директорию для модели
        model_path = os.path.join(self.model_dir, model_name)
        os.makedirs(model_path, exist_ok=True)
        
        # Сохраняем нейронную сеть
        self.model.save(os.path.join(model_path, 'neural_network.h5'))
        
        # Сохраняем scaler
        joblib.dump(self.scaler, os.path.join(model_path, 'scaler.pkl'))
        
        # Сохраняем label encoders
        joblib.dump(self.label_encoders, os.path.join(model_path, 'label_encoders.pkl'))
        
        # Сохраняем метрики
        if metrics:
            self.metrics_history[model_name] = metrics
            self.save_metrics_history()
            
            # Проверяем, является ли эта модель лучшей по MAE
            if metrics.get('mae', float('inf')) < self.best_mae:
                self.best_mae = metrics['mae']
                best_model_path = os.path.join(self.best_models_dir, 'best_model')
                
                # Удаляем предыдущую лучшую модель, если она существует
                if os.path.exists(best_model_path):
                    shutil.rmtree(best_model_path)
                
                # Копируем новую лучшую модель
                shutil.copytree(model_path, best_model_path)
                print(f"Новая лучшая модель! MAE: {self.best_mae:.2f}")
        
        print(f"Модель сохранена в директории: {model_path}")
        return model_path

    def load_model(self, model_path):
        """Загрузка модели и всех preprocessors"""
        # Загружаем нейронную сеть
        self.model = load_model(os.path.join(model_path, 'neural_network.h5'))
        
        # Загружаем scaler
        self.scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
        
        # Загружаем label encoders
        self.label_encoders = joblib.load(os.path.join(model_path, 'label_encoders.pkl'))
        
        print(f"Модель успешно загружена из: {model_path}")

    def update_model(self, new_data, epochs=10):
        """Обновление модели на новых данных"""
        # Предобработка новых данных
        X_train, X_test, y_train, y_test, _ = self.prepare_data(new_data)
        
        # Дообучение модели
        history = self.train(X_train, y_train, epochs=epochs)
        
        # Оценка на тестовой выборке
        loss, mae = self.evaluate(X_test, y_test)
        print(f'Ошибка после обновления: {loss:.2f}')
        print(f'MAE после обновления: {mae:.2f}')
        
        return history

    def get_best_model_info(self):
        """Получение информации о лучшей модели"""
        best_metrics = None
        best_model_name = None
        best_mae = float('inf')
        
        for model_name, metrics in self.metrics_history.items():
            if metrics['mae'] < best_mae:
                best_mae = metrics['mae']
                best_metrics = metrics
                best_model_name = model_name
        
        return best_model_name, best_metrics

    def estimate_price(self, apartment_data):
        """
        Оценка стоимости квартиры
        
        Параметры:
        apartment_data: dict - словарь с характеристиками квартиры
        Пример:
        {
            'Apartment type': 'Студия',
            'Metro station': 'Центральная',
            'Minutes to metro': 5,
            'Region': 'Центральный',
            'Number of rooms': 1,
            'Area': 40,
            'Living area': 35,
            'Kitchen area': 5,
            'Floor': 5,
            'Number of floors': 9,
            'Renovation': 'Косметический'
        }
        """
        # Преобразуем словарь в DataFrame
        df = pd.DataFrame([apartment_data])
        
        # Делаем предсказание
        predicted_price = self.predict(df)
        
        return float(predicted_price[0][0])

def main():
    # Загрузка данных
    data = pd.read_csv('data.csv')
    
    # Создание и обучение модели
    predictor = RealEstatePredictor()
    if os.path.exists(os.getcwd()+'models'):
        # Подготовка данных
        X_train, X_test, y_train, y_test, input_dim = predictor.prepare_data(data)
        
        # Создание модели
        predictor.create_model(input_dim)
        
        # Обучение модели
        history = predictor.train(X_train, y_train, epochs=100)
        
        # Оценка модели
        loss, mae = predictor.evaluate(X_test, y_test)
        print(f'Ошибка на тестовой выборке: {loss:.2f}')
        print(f'MAE на тестовой выборке: {mae:.2f}')
        
        # Визуализация процесса обучения
        predictor.plot_training_history(history)
    
    # Вывод информации о лучшей модели
    best_model_name, best_metrics = predictor.get_best_model_info()
    if best_metrics:
        print("\nИнформация о лучшей модели:")
        print(f"Имя модели: {best_model_name}")
        print(f"MAE: {best_metrics['mae']:.2f}")
        print(f"Дата создания: {best_metrics['timestamp']}")

    # Пример использования для оценки стоимости
    
    # Загружаем лучшую модель
    try:
        predictor.load_model('models/best_models/best_model')
    except:
        print("Лучшая модель не найдена. Сначала нужно обучить модель.")
        return
    
    # Пример квартиры для оценки
    test_apartment = {
        'Apartment type': 'Студия',
        'Metro station': 'Девяткино',
        'Minutes to metro': 25,
        'Region': 'Северо-Западный',
        'Number of rooms': 1,
        'Area': 23,
        'Living area': 18,
        'Kitchen area': 5,
        'Floor': 2,
        'Number of floors': 25,
        'Renovation': 'Косметический'
    }
    
    # Получаем оценку стоимости
    estimated_price = predictor.estimate_price(test_apartment)
    
    print("\nОценка стоимости квартиры:")
    print("============================")
    print("Характеристики квартиры:")
    for key, value in test_apartment.items():
        print(f"{key}: {value}")
    print("\nПредполагаемая стоимость: {:,.2f} руб.".format(estimated_price))

if __name__ == "__main__":
    main()