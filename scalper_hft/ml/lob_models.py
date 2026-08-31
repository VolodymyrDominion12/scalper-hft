"""Advanced ML Models for Limit Order Book (LOB) Data.

Реалізація архітектур глибокого навчання (Deep Learning) для
прогнозування напрямку ціни (або імбалансу) на основі L2 стакану.

Використовує PyTorch. За основу взято ідеї з "DeepLOB" (Zhang et al., 2019)
та "Machine Learning Algorithms with Applications in Finance" (Dixon).

Увага: для використання потрібен встановлений `torch`.
"""

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None  # type: ignore[assignment]

    class _NNMock:
        Module = object

    nn = _NNMock  # type: ignore[assignment]



class DeepLOB(nn.Module):
    """Спрощена версія DeepLOB: CNN + LSTM для L2 даних.

    Очікуваний вхідний тензор: (batch_size, 1, seq_len, features)
    features зазвичай 40 (10 рівнів bid_px, bid_vol, ask_px, ask_vol).
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()

        # Convolutional blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(1, 10)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(4, 1)),
            nn.LeakyReLU(negative_slope=0.01),
        )

        # Inception Module
        self.inp1 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(negative_slope=0.01),
        )

        # LSTM layer
        self.lstm = nn.LSTM(input_size=96, hidden_size=64, num_layers=1, batch_first=True)

        # Fully connected layer
        self.fc1 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x.shape: (batch, 1, seq_len, 40)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x_inp1 = self.inp1(x)
        x_inp2 = self.inp2(x)
        x_inp3 = self.inp3(x)

        x = torch.cat((x_inp1, x_inp2, x_inp3), dim=1)

        # (batch, 96, seq_len', 1) -> (batch, seq_len', 96)
        x = x.squeeze(-1).permute(0, 2, 1)

        x, _ = self.lstm(x)

        # Take the output of the last LSTM cell
        x = x[:, -1, :]
        x = self.fc1(x)

        return x


def prepare_lob_tensors(df, seq_len=100, num_classes=3):
    """Utility для підготовки даних (sliding window).
    Потребує pandas DataFrame із LOB-рівнями.
    """
    pass  # Реалізація залежить від конкретного датасету L2 (Tardis тощо)
