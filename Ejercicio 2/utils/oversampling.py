import numpy as np
from sklearn.utils import resample
from sklearn.utils import shuffle

def Oversampling(X, y, n_max):
    # Separar las clases
    X_0 = X[y == 0]
    X_1 = X[y == 1]

    X_0_upsampled = resample(X_0, replace=True, n_samples=n_max, random_state=42)
    y_0_upsampled = np.zeros(n_max)

    X_1_upsampled = resample(X_1, replace=True, n_samples=n_max, random_state=42)
    y_1_upsampled = np.ones(n_max)

    # Combinar y mezclar
    X_balanced = np.vstack([X_0_upsampled, X_1_upsampled])
    y_balanced = np.concatenate([y_0_upsampled, y_1_upsampled])

    X_balanced, y_balanced = shuffle(X_balanced, y_balanced, random_state=42)

    return X_balanced, y_balanced