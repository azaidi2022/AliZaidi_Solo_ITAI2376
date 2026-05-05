import torch
import torch.nn as nn


class CNNPageClassifier(nn.Module):
    """
    CNN page classifier for ruling relevance.

    Output:
    0 = relevant/content
    1 = non-content/other
    """

    def __init__(self):
        super(CNNPageClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),

            # Forces output to fixed size, avoids shape crashes
            nn.AdaptiveAvgPool1d(50)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 50, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x