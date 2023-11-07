"""Module providing a function modeling neural network."""
import torch
from torch import nn
from torchaudio.models import Conformer


class Classifier(nn.Module):
    """Speaker Classifier neural network"""

    def __init__(self, d_model=256, n_spks=600, dropout=0.1):
        """
        Initialize a Speaker Classifier neural network.

        Args:
            d_model (int): Dimension of the model's hidden layers (default is 256).
            n_spks (int): Number of speaker classes to classify (default is 600).
            dropout (float): Dropout rate for regularization (default is 0.1).

        The Classifier consists of a prenet, a Conformer-based feature extractor,
        and a prediction layer for speaker classification.

        - The prenet projects input features to the d_model dimension.
        - The Conformer processes the features to extract relevant information.
        - The prediction layer outputs speaker classification scores.

        The forward method accepts Mel spectrogram input and returns speaker scores.

        Args:
            mels (Tensor): Mel spectrogram input of shape (batch size, length, 40).

        Returns:
            out (Tensor): Speaker classification scores of shape (batch size, n_spks).
        """
        super().__init__()
        # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model)

        # transformer implementation
        # self.encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=d_model, nhead=128, dim_feedforward=2048, dropout=dropout
        # )
        # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        # end of transformer implementation

        # conformer implementation
        self.conformer = Conformer(
            input_dim=d_model,
            num_heads=32,  # num_heads=128
            ffn_dim=2048,  # ffn_dim=2048
            num_layers=3,   # num_layers=2
            depthwise_conv_kernel_size=31,  # depthwise_conv_kernel_size=31,
            dropout=dropout  # dropout
        )
        # end of conformer implementation

        # Project the the dimension of features from d_model into speaker nums.
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, n_spks),
        )

    def forward(self, mels):
        """
        Forward pass of the Speaker Classifier.

        Args:
            mels (Tensor): Mel spectrogram input of shape (batch size, length, 40).

        Returns:
            out (Tensor): Speaker classification scores of shape (batch size, n_spks).
        """
        # out: (batch size, length, d_model)
        out = self.prenet(mels)

        # transformer implementation
        # # out: (length, batch size, d_model)
        # out = out.permute(1, 0, 2)
        # # The encoder layer expect features in the shape of (length, batch size, d_model).
        # out = self.encoder_layer(out)
        # # out: (batch size, length, d_model)
        # out = out.transpose(0, 1)
        # end of transformer implementation

        # conformer implementation
        # length: each batch element has a scalar value equal to the length
        length = torch.full((out.size(0),), out.size(1), device=out.device)
        # out: (batch size, length, d_model)
        out, _ = self.conformer(out, length)
        # end of conformer implementation

        # mean pooling
        stats = out.mean(dim=1)

        # out: (batch, n_spks)
        out = self.pred_layer(stats)
        return out
