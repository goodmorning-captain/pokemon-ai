import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dataset.vgc_dataset import (  # noqa: E402
    VgcTeamDataset,
    load_teams,
    load_vocab,
    split_by_tournament,
    vocab_sizes,
)
from model.encoder import VgcTeamEncoder, infonce_loss  # noqa: E402

CHECKPOINT_DIR = os.path.join(ROOT, 'checkpoints')


def pick_device(name):
    if name == 'cpu':
        return torch.device('cpu')
    if name == 'cuda':
        return torch.device('cuda')
    if name == 'mps':
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def run_epoch(model, loader, optimizer, device, temperature, train, max_steps=None):
    model.train(train)
    total = 0.0
    n_batches = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for step, (view1, view2) in enumerate(tqdm(loader, disable=not train)):
            if max_steps is not None and step >= max_steps:
                break
            view1 = move_batch(view1, device)
            view2 = move_batch(view2, device)
            _, z1 = model(view1)
            _, z2 = model(view2)
            loss = infonce_loss(z1, z2, temperature=temperature)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += float(loss.item())
            n_batches += 1
    return total / max(n_batches, 1)


def save_checkpoint(path, model, optimizer, epoch, val_loss, args):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'val_loss': val_loss,
            'args': vars(args),
        },
        path,
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Train VGC team encoder with InfoNCE')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--val-frac', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--poke-dim', type=int, default=64)
    parser.add_argument('--team-dim', type=int, default=128)
    parser.add_argument('--proj-dim', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--max-steps', type=int, default=None, help='cap train batches per epoch (smoke test)')
    parser.add_argument('--checkpoint-dir', type=str, default=CHECKPOINT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    vocab = load_vocab()
    teams = load_teams()
    sizes = vocab_sizes(vocab)
    train_idx, val_idx, val_ids = split_by_tournament(
        teams['tournament_id'], val_frac=args.val_frac, seed=args.seed
    )

    train_data = VgcTeamDataset(teams, indices=train_idx, augment=True)
    val_data = VgcTeamDataset(teams, indices=val_idx, augment=True)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model = VgcTeamEncoder(
        sizes,
        poke_dim=args.poke_dim,
        team_dim=args.team_dim,
        proj_dim=args.proj_dim,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f'device={device}')
    print(f'train teams={len(train_data)} val teams={len(val_data)} val tournaments={len(val_ids)}')
    print(f'vocab sizes={sizes}')

    best_val = float('inf')
    last_path = os.path.join(args.checkpoint_dir, 'last.pt')
    best_path = os.path.join(args.checkpoint_dir, 'best.pt')

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer, device, args.temperature, train=True, max_steps=args.max_steps
        )
        if args.max_steps is not None:
            print(f'epoch {epoch:03d}  train_nce={train_loss:.4f}  (val skipped; --max-steps)')
            save_checkpoint(last_path, model, optimizer, epoch, train_loss, args)
            continue

        val_loss = run_epoch(
            model, val_loader, optimizer, device, args.temperature, train=False
        )
        print(f'epoch {epoch:03d}  train_nce={train_loss:.4f}  val_nce={val_loss:.4f}')
        save_checkpoint(last_path, model, optimizer, epoch, val_loss, args)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(best_path, model, optimizer, epoch, val_loss, args)
            print(f'  saved best -> {best_path}')


if __name__ == '__main__':
    main()
