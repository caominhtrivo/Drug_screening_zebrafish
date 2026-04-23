# src/models/architectures.py
import torch
import torch.nn as nn

class ZebrafishLSTM(nn.Module):
    """
    Baseline LSTM model for sequence classification.
    Uses a Linear layer for dimensionality reduction followed by a standard LSTM.
    """
    def __init__(self, input_dim=4096, hidden_dim=256, num_layers=2, num_classes=16):
        super(ZebrafishLSTM, self).__init__()
        # Feature reduction before LSTM to handle high-dimensional ResNet features
        self.feature_reducer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.lstm = nn.LSTM(512, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Apply Gaussian noise as a regularizer during training
        if self.training:
            x = x + torch.randn_like(x) * 0.01

        x = self.feature_reducer(x)
        lstm_out, _ = self.lstm(x)
        # Sequence-to-Label: take the hidden state of the final time step
        last_out = lstm_out[:, -1, :] 
        return self.classifier(last_out)


class ConvBiLSTM(nn.Module):
    """
    Advanced Hybrid Architecture: 1D-CNN + Bidirectional LSTM.
    Captures local temporal patterns via CNN and global dependencies via Bi-LSTM.
    """
    def __init__(self, input_dim=4096, hidden_dim=256, num_layers=2, num_classes=16):
        super(ConvBiLSTM, self).__init__()

        # 1D-CNN: Extracts local motion features (e.g., rapid bursts or small vibrations)
        # Input expected: [batch, input_dim, seq_len]
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Bi-LSTM: Learns global long-term temporal dynamics of swimming behavior
        self.lstm = nn.LSTM(512, hidden_dim, num_layers, 
                            batch_first=True, bidirectional=True, dropout=0.3)

        # Final Classifier for compound identification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256), # *2 due to Bidirectional concatenation
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        # Transpose for Conv1d: [batch_size, input_dim, seq_len]
        x = x.permute(0, 2, 1)

        # Local feature extraction
        x = self.conv1d(x)

        # Prepare for LSTM: [batch_size, seq_len, 512]
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)

        # Global Average Pooling over the temporal dimension for robustness
        pooled_out = lstm_out.mean(dim=1)

        return self.classifier(pooled_out)