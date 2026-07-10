import torch
import torch.nn as nn
from panns_inference.models import Cnn14
from pathlib import Path

# Keeping your exact original model checkpoint path
DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parent.parent / "model" / "Cnn14_emb512_mAP=0.420.pth"
)


class PannGru(nn.Module):
    def __init__(
        self, hidden_size: int = 128, checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH
    ):
        super(PannGru, self).__init__()

        base_model = Cnn14(
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=527,
        )
        base_model.fc1 = nn.Linear(2048, 512)
        base_model.fc_audioset = nn.Linear(512, 527)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        base_model.load_state_dict(checkpoint["model"])
        self.pann = base_model

        # PANN will not change
        for p in self.pann.parameters():
            p.requires_grad_(False)
        self.pann.eval()

        self.gru = nn.GRU(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.20,
            bias=False,
        )

        self.projection_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )

    def train(self, mode: bool = True):
        """Guarantees the backbone's BatchNorm/Dropout statistics never update."""
        super().train(mode)
        self.pann.eval()
        return self

    def forward(
        self, waveform: torch.Tensor, chunk_duration_samples: int = 80000
    ) -> torch.Tensor:
        """
        Args:
            waveform: Tensor of shape [batch_size, total_samples] (Expected incoming SR: 16000)
            chunk_duration_samples: Length of each incoming temporal chunk at 16kHz.
                                    Default 80000 samples = 5 seconds at 16kHz.
        """
        batch_size, total_samples = waveform.shape

        chunks = list(waveform.split(chunk_duration_samples, dim=1))

        # Handle trailing variations with zero padding
        if chunks[-1].shape[1] < chunk_duration_samples:
            padding = chunk_duration_samples - chunks[-1].shape[1]
            chunks[-1] = nn.functional.pad(chunks[-1], (0, padding))

        num_chunks = len(chunks)

        # Reshape to [batch_size, num_chunks, chunk_duration_samples]
        chunked_waveforms = torch.stack(chunks, dim=1)

        # Flatten
        flat_chunks = chunked_waveforms.view(-1, chunk_duration_samples)

        # linear interpolation from 16kHz to 32kHz
        # Adds an explicit channel dimension for interpolate, then squeezes it back.
        # Target samples = 5 seconds * 32000 Hz = 160000 samples
        flat_chunks_32k = nn.functional.interpolate(
            flat_chunks.unsqueeze(1), size=160000, mode="linear", align_corners=True
        ).squeeze(1)

        # Extract Features using 32kHz Pretrained PANN
        with torch.no_grad():
            pann_out = self.pann(flat_chunks_32k)
            flat_embeddings = pann_out["embedding"]

        # Reconstruct Temporal Sequence for GRU [batch_size, num_chunks, 512]
        features = flat_embeddings.view(batch_size, num_chunks, 512)

        # GRU & Deep SVDD Projection
        gru_out, _ = self.gru(features)
        last_time_frame = gru_out[:, -1, :]  # Final temporal sequence step

        final_embedding = self.projection_head(last_time_frame)
        return final_embedding
