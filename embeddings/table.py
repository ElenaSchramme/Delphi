import torch
import torch.nn as nn
import pandas as pd
import numpy as np

"""So far the same as linear embedding. Will be adjusted with code from this ressource: https://github.com/lucidrains/tab-transformer-pytorch/blob/main/tab_transformer_pytorch/ft_transformer.py"""

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

class TableEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super(TableEmbedding, self).__init__()
        self.linear = nn.Linear(input_dim, embed_dim)
    
    def forward(self, x):
        return self.linear(x)

input_dim = matrix_data.shape[1]  # Anzahl der Spalten
embed_dim = 128  # Beispiel für die Embedding-Größe
table_embed = TableEmbedding(matrix_data.shape[1], embed_dim)

# Beispiel: Embedding der gesamten Tabelle
embedded_table_uniform = table_embed(matrix_data)
print(embedded_table_uniform)
