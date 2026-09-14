import json
import os
import random

import torch
from torch.utils.data import Dataset

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
TEAMS_PATH = os.path.join(PROCESSED_DIR, 'teams.pt')
VOCAB_PATH = os.path.join(PROCESSED_DIR, 'vocab.json')

PAD = 0
TEAM_FIELDS = ('species', 'item', 'ability', 'tera', 'moves')


def load_vocab(path=VOCAB_PATH):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_teams(path=TEAMS_PATH):
    return torch.load(path, weights_only=False)


def vocab_sizes(vocab):
    return {
        'species': len(vocab['species']),
        'item': len(vocab['item']),
        'ability': len(vocab['ability']),
        'tera': len(vocab['tera']),
        'move': len(vocab['move']),
    }


def split_by_tournament(tournament_ids, val_frac=0.15, seed=0):
    unique = sorted(set(tournament_ids))
    rng = random.Random(seed)
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * val_frac)))
    n_val = min(n_val, len(unique) - 1) if len(unique) > 1 else 1
    val_ids = set(unique[:n_val])
    train_idx = [i for i, tid in enumerate(tournament_ids) if tid not in val_ids]
    val_idx = [i for i, tid in enumerate(tournament_ids) if tid in val_ids]
    return train_idx, val_idx, val_ids


def augment_team(team, pad=PAD):
    species = team['species'].clone()
    item = team['item'].clone()
    ability = team['ability'].clone()
    tera = team['tera'].clone()
    moves = team['moves'].clone()

    valid = (species != pad).nonzero(as_tuple=True)[0]
    if valid.numel() == 0:
        return {
            'species': species,
            'item': item,
            'ability': ability,
            'tera': tera,
            'moves': moves,
        }

    slot = int(valid[int(torch.randint(0, valid.numel(), (1,)).item())].item())
    action = int(torch.randint(0, 3, (1,)).item())

    if action == 0 and valid.numel() > 1:
        species[slot] = pad
        item[slot] = pad
        ability[slot] = pad
        tera[slot] = pad
        moves[slot] = pad
    elif action == 1:
        item[slot] = pad
    else:
        move_valid = (moves[slot] != pad).nonzero(as_tuple=True)[0]
        if move_valid.numel() > 0:
            mi = int(move_valid[int(torch.randint(0, move_valid.numel(), (1,)).item())].item())
            moves[slot, mi] = pad
        elif valid.numel() > 1:
            species[slot] = pad
            item[slot] = pad
            ability[slot] = pad
            tera[slot] = pad
            moves[slot] = pad
        else:
            item[slot] = pad

    return {
        'species': species,
        'item': item,
        'ability': ability,
        'tera': tera,
        'moves': moves,
    }


class VgcTeamDataset(Dataset):
    """Yields two views of a team for InfoNCE. Does not load gen9.json."""

    def __init__(self, teams, indices=None, augment=True):
        self.augment = augment
        n = teams['species'].shape[0]
        self.indices = list(range(n) if indices is None else indices)
        self.species = teams['species']
        self.item = teams['item']
        self.ability = teams['ability']
        self.tera = teams['tera']
        self.moves = teams['moves']

    def __len__(self):
        return len(self.indices)

    def _team(self, i):
        j = self.indices[i]
        return {
            'species': self.species[j],
            'item': self.item[j],
            'ability': self.ability[j],
            'tera': self.tera[j],
            'moves': self.moves[j],
        }

    def __getitem__(self, i):
        team = self._team(i)
        if self.augment:
            return augment_team(team), augment_team(team)
        return team, team
