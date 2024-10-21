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

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, input_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

input_dim = matrix_data.shape[1]  # Anzahl der Spalten
embed_dim = 128  # Beispiel für die Embedding-Größe
autoencoder = AutoEncoder(input_dim, embed_dim)

# Beispiel: Encoding der Matrix
encoded_matrix, decoded_matrix = autoencoder(matrix_data)
print(encoded_matrix)   
print(decoded_matrix)
