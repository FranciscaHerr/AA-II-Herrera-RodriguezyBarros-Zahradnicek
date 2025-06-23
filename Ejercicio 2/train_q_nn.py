import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle as shuffle_data
from sklearn.preprocessing import StandardScaler

# --- Constantes y Configuración ---
MODEL_SAVE_PATH = 'flappy_birds_q_expert_model.h5'

# Hiperparámetros de entrenamiento
EPOCHS = 1000
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42 

# --- Cargar Q-table entrenada ---
QTABLE_PATH = 'flappy_birds_q_table_final.pkl'  # Cambia el path si es necesario
with open(QTABLE_PATH, 'rb') as f:
    q_table = pickle.load(f)

# --- Preparar datos para entrenamiento ---
# Convertir la Q-table en X (estados) e y (valores Q para cada acción)
X = []  # Estados discretos
y = []  # Q-values para cada acción
for state, q_values in q_table.items():
    X.append(state)
    y.append(np.argmax(q_values))
X = np.array(X)
y = np.array(y)
y = 1 - y

from utils import Oversampling
X, y = Oversampling(X, y, 12000)

print(f"\nDividiendo datos en entrenamiento y validación (split: {VALIDATION_SPLIT})...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=VALIDATION_SPLIT,
    random_state=RANDOM_STATE,
)
print(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]}")
print(f"Tamaño del conjunto de validación: {X_val.shape[0]}")

print("\nDefiniendo el modelo de red neuronal...")
num_features = X.shape[1]
# --- Definir la red neuronal ---
model = Sequential([
    Input(shape=(num_features,)),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# ReduceLROnPlateau
reduce_lr_callback = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5,         
    patience=10,         
    min_lr=1e-6,      
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=50,
    restore_best_weights=True,
    verbose=1
)

callbacks_list = [reduce_lr_callback, early_stop]

print(f"\nEntrenando el modelo durante {EPOCHS} épocas con batch_size {BATCH_SIZE}...")
try:
    history = model.fit(X_train, y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        validation_data=(X_val, y_val),
                        callbacks=callbacks_list,
                        verbose=1)
except KeyboardInterrupt:
    pass

# --- Evaluación y Guardado del Modelo ---
print("\nEntrenamiento completado.")

print(f"\nGuardando el modelo entrenado en: {MODEL_SAVE_PATH}")
model.save(MODEL_SAVE_PATH)
print("Modelo guardado exitosamente.")

import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.grid(True)
plt.title('Binary Crossentropy')
plt.show()

plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.grid(True)
plt.title('Accuracy')
plt.show()