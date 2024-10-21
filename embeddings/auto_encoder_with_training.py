import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Definiere den Auto-Encoder
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

# Instanziiere den Auto-Encoder
input_dim = matrix_data.shape[1]  # Anzahl der Spalten
embed_dim = 8  # needs to be smaller than input_dim
autoencoder = AutoEncoder(input_dim, embed_dim)

# Definiere den Optimierer und die Verlustfunktion
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)  # Adam Optimizer

# Training des Auto-Encoders
num_epochs = 1000
for epoch in range(num_epochs):
    # Vorwärts-Pass: Berechne die Encodings und Rekonstruktionen
    encoded, decoded = autoencoder(matrix_data)
    
    # Berechne den Verlust (wie gut die Rekonstruktion ist)
    loss = criterion(decoded, matrix_data)
    
    # Rückwärts-Pass: Berechne die Gradienten und optimiere die Gewichte
    optimizer.zero_grad()  # Setzt die Gradienten zurück
    loss.backward()        # Berechnet die Gradienten
    optimizer.step()       # Aktualisiert die Gewichte
    
    # Gelegentliche Ausgabe des Verlusts, um den Fortschritt zu überwachen
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Nach dem Training kannst du den Auto-Encoder verwenden, um die Daten zu codieren und zu rekonstruieren
encoded_matrix, decoded_matrix = autoencoder(matrix_data)

"""print("\nEncoded matrix:")
print(encoded_matrix)

print("\nDecoded matrix (Reconstructed):")
print(decoded_matrix)"""
