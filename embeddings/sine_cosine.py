import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Datei einlesen
file_path = 'embeddings/mock_up_data.txt'

# Daten aus der txt-Datei laden
df = pd.read_csv(file_path, sep='\s+')  # Verwende Leerzeichen als Trennzeichen

# Fehlende Werte (NA) durch Nullen ersetzen
df = df.apply(pd.to_numeric, errors='coerce').fillna(0)  # Konvertiere 'NA' zu NaN und ersetze NaN durch 0

# Entferne die Spaltenüberschriften und die ID-Spalte
data = df.iloc[:, 1:].values  # Exkludiere die erste Spalte (IDs)

# Konvertiere die Daten in einen PyTorch Tensor
matrix_data = torch.tensor(data, dtype=torch.float32)

# Ausgabe des Tensors
print(matrix_data)

def sine_cosine_embedding(matrix, embed_dim):
    result = []
    for i in range(matrix.shape[1]):  # Für jede Spalte
        position = np.arange(matrix.shape[0])
        div_term = np.exp(np.arange(0, embed_dim, 2) * -(np.log(10000.0) / embed_dim))
        sinusoidal_embed = np.zeros((matrix.shape[0], embed_dim))
        sinusoidal_embed[:, 0::2] = np.sin(position[:, None] * div_term)
        sinusoidal_embed[:, 1::2] = np.cos(position[:, None] * div_term)
        result.append(torch.tensor(sinusoidal_embed, dtype=torch.float32))
    return torch.stack(result).sum(dim=0)  # Summiere die Ergebnisse der Spalten

# Beispiel: Sine-/Cosine-Embedding

input_dim = matrix_data.shape[1]  # Anzahl der Spalten
embed_dim = 128  # Beispiel für die Embedding-Größe
sine_cosine_embedded = sine_cosine_embedding(matrix_data.numpy(), embed_dim)

print(sine_cosine_embedded)
