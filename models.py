import torch
import torch.nn as nn

class AttentionCNN(nn.Module):
    def __init__(self, input_length=448):
        super(AttentionCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(2)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=16)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (input_length // 8), 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        x = x.permute(2, 0, 1)  # Reshape for attention
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output  # Residual connection
        x = x.permute(1, 2, 0)  # Reshape back
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x



# Define SpectralCNN (simplified, no batch norm)
class SpectralCNN(nn.Module):
    def __init__(self, input_length=448):
        super(SpectralCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * (input_length // 8), 128)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    

# Define Transformer Model
# class TransformerCNN(nn.Module):
#     def __init__(self, input_length=448, d_model=64, n_heads=8, n_layers=2):
#         super(TransformerCNN, self).__init__()
#         self.input_length = input_length
#         self.embedding = nn.Conv1d(1, d_model, kernel_size=5, padding=2)
#         encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
#         self.pool = nn.AdaptiveAvgPool1d(1)
#         self.fc1 = nn.Linear(d_model, 128)
#         self.dropout = nn.Dropout(0.4)
#         self.fc2 = nn.Linear(128, 1)

#     def forward(self, x):
#         x = torch.relu(self.embedding(x))
#         x = x.permute(0, 2, 1)  # (batch, seq_len, d_model)
#         x = self.transformer(x)
#         x = x.permute(0, 2, 1)  # (batch, d_model, seq_len)
#         x = self.pool(x).squeeze(-1)
#         x = torch.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x