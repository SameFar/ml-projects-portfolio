import torch
import librosa
from torch.utils.data import TensorDataset, DataLoader
import logging
from pathlib import Path
import numpy as np


def preprocess_single_audio(file_path: Path) -> torch.Tensor:

    # Loads audiofile, specifying mono sound and sample rate f 16khz
    y, _ = librosa.load(str(file_path), sr=16000, mono=True)

    # Force 30sec duration
    target_samples = 16000 * 30
    current_samples = y.shape[0]

    # Clamp if too large
    if current_samples > target_samples:
        y = y[:target_samples]

    # Pad with silence if too small
    elif current_samples < target_samples:
        pad_amount = target_samples - current_samples
        y = np.pad(y, (0, pad_amount), mode="constant")

    # Convert back into a 1D PyTorch float tensor
    return torch.from_numpy(y).float()


def make_dataloader(songs_dir: Path):
    audio_files = list(songs_dir.glob("*.mp3"))

    logging.info(f"Preprocessing {len(audio_files)} songs...")
    all_processed_waveforms = []

    for file_path in audio_files:
        try:
            clean_tensor = preprocess_single_audio(file_path)
            all_processed_waveforms.append(clean_tensor)
        except Exception as e:
            logging.error(f"Skipping corrupt file {file_path}: {e}")

    # Fallback safety validation check to avoid runtime torch.stack crashes
    if not all_processed_waveforms:
        raise ValueError(
            f"No valid audio tracks were successfully preprocessed inside: {songs_dir}"
        )

    dataset_matrix = torch.stack(all_processed_waveforms)
    dataset = TensorDataset(dataset_matrix)

    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    logging.info("DataLoader initialized successfully!")

    return loader
