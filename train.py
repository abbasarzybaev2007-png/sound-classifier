"""Train with folds 1-3, select on 4, and evaluate once on 5."""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from src.data import ESC10, read_metadata
from src.model import SoundCNN


def evaluate(model, loader, device):
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            predictions.extend(model(x.to(device)).argmax(1).cpu().tolist())
            targets.extend(y.tolist())
    return targets, predictions


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=Path('data/ESC-50'))
    p.add_argument('--output', type=Path, default=Path('checkpoints/best.pt'))
    p.add_argument('--results', type=Path, default=Path('results/test.json'))
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--noise', type=float, default=0.0, help='Training noise standard deviation')
    p.add_argument('--test-noise', type=float, default=0.0, help='Test noise standard deviation')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or min(args.noise, args.test_noise) < 0:
        p.error('epochs and batch-size must be positive; noise must be nonnegative')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, labels = read_metadata(args.data)
    def loader(folds, noise=0., shuffle=False):
        return DataLoader(ESC10(args.data, folds, labels, noise, args.seed),
                          batch_size=args.batch_size, shuffle=shuffle, num_workers=0)
    train = loader([1, 2, 3], args.noise, True)
    val, test = loader([4]), loader([5], args.test_noise)
    model = SoundCNN(len(labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.results.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for x, y in train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(); loss = loss_fn(model(x), y)
            loss.backward(); optimizer.step()
            total += loss.item() * len(y)
        truth, pred = evaluate(model, val, device)
        score = f1_score(truth, pred, average='macro', zero_division=0)
        print(f'epoch={epoch+1} loss={total/len(train.dataset):.4f} val_macro_f1={score:.4f}', flush=True)
        if score > best:
            best = score
            torch.save({'state_dict': model.state_dict(), 'labels': labels, 'epoch': epoch + 1}, args.output)
    checkpoint = torch.load(args.output, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    truth, pred = evaluate(model, test, device)
    result = {'accuracy': accuracy_score(truth, pred),
              'macro_f1': f1_score(truth, pred, average='macro', zero_division=0),
              'labels': labels, 'confusion_matrix': confusion_matrix(truth, pred, labels=list(range(len(labels)))).tolist(),
              'classification_report': classification_report(truth, pred, labels=list(range(len(labels))), target_names=labels, output_dict=True, zero_division=0),
              'train_folds': [1, 2, 3], 'validation_fold': 4, 'test_fold': 5,
              'train_noise': args.noise, 'test_noise': args.test_noise, 'seed': args.seed,
              'selected_epoch': checkpoint['epoch']}
    args.results.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(f"Test accuracy={result['accuracy']:.4f}; macro-F1={result['macro_f1']:.4f}")


if __name__ == '__main__':
    main()
