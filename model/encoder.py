import torch
import torch.nn as nn
import torch.nn.functional as F

PAD = 0


class PokemonEncoder(nn.Module):
    """Embed species/item/ability/tera/moves, then MLP to one vector per slot."""

    def __init__(
        self,
        sizes,
        species_dim=32,
        item_dim=16,
        ability_dim=16,
        tera_dim=8,
        move_dim=16,
        poke_dim=64,
    ):
        super().__init__()
        self.species = nn.Embedding(sizes['species'], species_dim, padding_idx=PAD)
        self.item = nn.Embedding(sizes['item'], item_dim, padding_idx=PAD)
        self.ability = nn.Embedding(sizes['ability'], ability_dim, padding_idx=PAD)
        self.tera = nn.Embedding(sizes['tera'], tera_dim, padding_idx=PAD)
        self.move = nn.Embedding(sizes['move'], move_dim, padding_idx=PAD)

        in_dim = species_dim + item_dim + ability_dim + tera_dim + move_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, poke_dim),
            nn.GELU(),
            nn.Linear(poke_dim, poke_dim),
        )

    def forward(self, species, item, ability, tera, moves):
        species_e = self.species(species)
        item_e = self.item(item)
        ability_e = self.ability(ability)
        tera_e = self.tera(tera)

        move_e = self.move(moves)
        move_mask = (moves != PAD).unsqueeze(-1).float()
        move_sum = (move_e * move_mask).sum(dim=2)
        move_count = move_mask.sum(dim=2).clamp(min=1.0)
        move_pool = move_sum / move_count

        cat = torch.cat([species_e, item_e, ability_e, tera_e, move_pool], dim=-1)
        poke_h = self.mlp(cat)

        mask = species != PAD
        poke_h = poke_h * mask.unsqueeze(-1).float()
        return poke_h, mask


class TeamEncoder(nn.Module):
    """Order-invariant mean over non-PAD slots, then MLP."""

    def __init__(self, poke_dim, team_dim=128):
        super().__init__()
        self.out = nn.Sequential(
            nn.LayerNorm(poke_dim),
            nn.Linear(poke_dim, team_dim),
            nn.GELU(),
            nn.Linear(team_dim, team_dim),
        )

    def forward(self, poke_h, mask):
        mask_f = mask.unsqueeze(-1).float()
        pooled = (poke_h * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return self.out(pooled)


class VgcTeamEncoder(nn.Module):
    """h is the team representation; z is the InfoNCE projection."""

    def __init__(self, sizes, poke_dim=64, team_dim=128, proj_dim=64):
        super().__init__()
        self.pokemon = PokemonEncoder(sizes, poke_dim=poke_dim)
        self.team = TeamEncoder(poke_dim, team_dim=team_dim)
        self.projector = nn.Sequential(
            nn.Linear(team_dim, team_dim),
            nn.GELU(),
            nn.Linear(team_dim, proj_dim),
        )

    def encode_team(self, batch):
        poke_h, mask = self.pokemon(
            batch['species'],
            batch['item'],
            batch['ability'],
            batch['tera'],
            batch['moves'],
        )
        return self.team(poke_h, mask)

    def forward(self, batch):
        h = self.encode_team(batch)
        z = self.projector(h)
        return h, z


def infonce_loss(z1, z2, temperature=0.07):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    loss_ab = F.cross_entropy(logits, labels)
    loss_ba = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_ab + loss_ba)
