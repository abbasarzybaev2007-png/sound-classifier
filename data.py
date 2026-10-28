"""ESC-10 loading and audio preprocessing shared by training and inference."""
from pathlib import Path
import csv

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

SAMPLE_RATE = 22050
SAMPLES = SAMPLE_RATE * 5


def read_audio(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=5.0)
    return librosa.util.fix_length(audio, size=SAMPLES).astype(np.float32)


def features(audio):
    mel = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_fft=1024,
                                        hop_length=512, n_mels=64, power=2)
    # A fixed reference keeps preprocessing identical across training and inference.
    db = librosa.power_to_db(mel, ref=1.0, top_db=80)
    return torch.from_numpy(((db + 40.0) / 40.0).astype(np.float32)).unsqueeze(0)


def read_metadata(root):
    root = Path(root)
    with (root / 'meta' / 'esc50.csv').open(encoding='utf-8', newline='') as f:
        rows = [row for row in csv.DictReader(f) if row['esc10'].lower() == 'true']
    labels = sorted({row['category'] for row in rows})
    if len(labels) != 10:
        raise ValueError(f'Expected 10 ESC-10 categories, found {len(labels)}')
    return rows, labels


class ESC10(Dataset):
    def __init__(self, root, folds, labels, noise=0.0, seed=42):
        self.root, self.labels = Path(root), labels
        rows, _ = read_metadata(root)
        self.rows = [r for r in rows if int(r['fold']) in folds]
        if not self.rows:
            raise ValueError('No recordings found for the selected folds')
        self.noise, self.rng = noise, np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        audio = read_audio(self.root / 'audio' / row['filename'])
        if self.noise:
            audio = audio + self.rng.normal(0, self.noise, audio.shape).astype(np.float32)
        return features(audio), self.labels.index(row['category'])
