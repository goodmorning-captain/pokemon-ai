import json
import os

import torch

SRC_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'gen9.json')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
VOCAB_PATH = os.path.join(OUT_DIR, 'vocab.json')
TEAMS_PATH = os.path.join(OUT_DIR, 'teams.pt')
MATCHES_PATH = os.path.join(OUT_DIR, 'matches.jsonl')

TEAM_SIZE = 6
MOVE_SLOTS = 4
PAD = 0
UNK = 1
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
MISSING_PLACING = -1


def norm_id(value):
    if value is None or value == '':
        return None
    return str(value)


def id_sort_key(pid):
    if pid.isdigit():
        return (0, int(pid))
    return (1, pid)


def build_stoi(values):
    stoi = {PAD_TOKEN: PAD, UNK_TOKEN: UNK}
    for value in sorted(values):
        if value in (None, '', PAD_TOKEN, UNK_TOKEN):
            continue
        if value not in stoi:
            stoi[value] = len(stoi)
    return stoi


def lookup(stoi, value):
    if value in (None, ''):
        return PAD
    return stoi.get(value, UNK)


def sorted_pokemon(decklist):
    rows = []
    for pokemon in decklist or []:
        pid = norm_id(pokemon.get('id'))
        if pid is None:
            continue
        badges = pokemon.get('badges') or []
        moves = [move for move in badges[:MOVE_SLOTS] if move]
        rows.append({
            'id': pid,
            'name': pokemon.get('name') or '',
            'item': pokemon.get('item') or '',
            'ability': pokemon.get('ability') or '',
            'tera': pokemon.get('teratype') or '',
            'moves': moves,
        })
    rows.sort(key=lambda row: id_sort_key(row['id']))
    return rows[:TEAM_SIZE]


def add_pokemon_to_vocab(pokemon, species_id_to_name, items, abilities, teras, moves):
    pid = norm_id(pokemon.get('id'))
    if pid is None:
        return
    name = pokemon.get('name') or ''
    if pid not in species_id_to_name and name:
        species_id_to_name[pid] = name
    item = pokemon.get('item')
    if item:
        items.add(item)
    ability = pokemon.get('ability')
    if ability:
        abilities.add(ability)
    tera = pokemon.get('teratype')
    if tera:
        teras.add(tera)
    for move in pokemon.get('badges') or []:
        if move:
            moves.add(move)


def collect_vocab(players):
    species_id_to_name = {}
    items = set()
    abilities = set()
    teras = set()
    moves = set()
    divisions = set()

    def walk_decklist(decklist):
        for pokemon in decklist or []:
            add_pokemon_to_vocab(
                pokemon, species_id_to_name, items, abilities, teras, moves
            )

    for player in players:
        division = player.get('division')
        if division:
            divisions.add(division)
        walk_decklist(player.get('decklist'))
        for opponent in player.get('opponents') or []:
            walk_decklist(opponent.get('decklist'))

    vocab = {
        'pad_id': PAD,
        'unk_id': UNK,
        'species_id_to_name': dict(sorted(species_id_to_name.items(), key=lambda kv: id_sort_key(kv[0]))),
        'species': build_stoi(species_id_to_name.keys()),
        'item': build_stoi(items),
        'ability': build_stoi(abilities),
        'tera': build_stoi(teras),
        'move': build_stoi(moves),
        'division': build_stoi(divisions),
    }
    return vocab


def encode_team(decklist, vocab):
    species = [PAD] * TEAM_SIZE
    item = [PAD] * TEAM_SIZE
    ability = [PAD] * TEAM_SIZE
    tera = [PAD] * TEAM_SIZE
    moves = [[PAD] * MOVE_SLOTS for _ in range(TEAM_SIZE)]

    for slot, pokemon in enumerate(sorted_pokemon(decklist)):
        species[slot] = lookup(vocab['species'], pokemon['id'])
        item[slot] = lookup(vocab['item'], pokemon['item'])
        ability[slot] = lookup(vocab['ability'], pokemon['ability'])
        tera[slot] = lookup(vocab['tera'], pokemon['tera'])
        for move_i, move in enumerate(pokemon['moves']):
            moves[slot][move_i] = lookup(vocab['move'], move)

    return species, item, ability, tera, moves


def build():
    if not os.path.isfile(SRC_PATH):
        raise FileNotFoundError(f'missing {os.path.abspath(SRC_PATH)}; run scraper/process_raw.py first')

    try:
        with open(SRC_PATH, 'r', encoding='utf-8') as f:
            players = json.load(f)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f'{SRC_PATH} is truncated or invalid JSON; re-run scraper/process_raw.py ({exc})'
        ) from exc

    vocab = collect_vocab(players)

    species_rows = []
    item_rows = []
    ability_rows = []
    tera_rows = []
    move_rows = []
    placings = []
    division_ids = []
    tournament_ids = []
    player_names = []
    team_index = {}

    skipped_teams = 0
    for player in players:
        decklist = player.get('decklist') or []
        if not decklist:
            skipped_teams += 1
            continue

        species, item, ability, tera, moves = encode_team(decklist, vocab)
        if all(value == PAD for value in species):
            skipped_teams += 1
            continue

        idx = len(species_rows)
        species_rows.append(species)
        item_rows.append(item)
        ability_rows.append(ability)
        tera_rows.append(tera)
        move_rows.append(moves)

        placing = player.get('placing')
        placings.append(MISSING_PLACING if placing is None else int(placing))
        division_ids.append(lookup(vocab['division'], player.get('division')))
        tournament_ids.append(str(player.get('tournament_id') or ''))
        player_names.append(player.get('name') or '')

        key = (tournament_ids[-1], player.get('division') or '', player_names[-1])
        if key[2] and key not in team_index:
            team_index[key] = idx

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(VOCAB_PATH, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, indent=2)

    payload = {
        'species': torch.tensor(species_rows, dtype=torch.long),
        'item': torch.tensor(item_rows, dtype=torch.long),
        'ability': torch.tensor(ability_rows, dtype=torch.long),
        'tera': torch.tensor(tera_rows, dtype=torch.long),
        'moves': torch.tensor(move_rows, dtype=torch.long),
        'placing': torch.tensor(placings, dtype=torch.long),
        'division_id': torch.tensor(division_ids, dtype=torch.long),
        'tournament_id': tournament_ids,
        'player_name': player_names,
    }
    torch.save(payload, TEAMS_PATH)

    n_matches = 0
    skipped_matches = 0
    with open(MATCHES_PATH, 'w', encoding='utf-8') as f:
        for player in players:
            tournament_id = str(player.get('tournament_id') or '')
            division = player.get('division') or ''
            name = player.get('name') or ''
            team_a = team_index.get((tournament_id, division, name))
            if team_a is None:
                continue

            for opponent in player.get('opponents') or []:
                if not opponent.get('decklist'):
                    continue
                opp_name = opponent.get('name') or ''
                team_b = team_index.get((tournament_id, division, opp_name))
                if team_b is None or team_b == team_a:
                    skipped_matches += 1
                    continue
                row = {
                    'team_idx_a': team_a,
                    'team_idx_b': team_b,
                    'result': opponent.get('result'),
                }
                f.write(json.dumps(row) + '\n')
                n_matches += 1

    return {
        'n_players': len(players),
        'n_teams': len(species_rows),
        'skipped_teams': skipped_teams,
        'n_matches': n_matches,
        'skipped_matches': skipped_matches,
        'vocab': vocab,
    }


def print_stats(stats):
    vocab = stats['vocab']
    print(f'teams: {stats["n_teams"]} (skipped {stats["skipped_teams"]}) from {stats["n_players"]} player rows')
    print(f'matches: {stats["n_matches"]} (skipped {stats["skipped_matches"]})')
    print('vocab sizes (including PAD/UNK):')
    for field in ('species', 'item', 'ability', 'tera', 'move', 'division'):
        print(f'  {field}: {len(vocab[field])}')
    print(f'  species_id_to_name: {len(vocab["species_id_to_name"])}')
    print(f'wrote {os.path.abspath(VOCAB_PATH)}')
    print(f'wrote {os.path.abspath(TEAMS_PATH)}')
    print(f'wrote {os.path.abspath(MATCHES_PATH)}')


if __name__ == '__main__':
    print_stats(build())