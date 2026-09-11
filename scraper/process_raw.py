import json
import os

RAW_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
OUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'gen9.json')
GEN9_ID_MIN = 20
GEN9_ID_MAX = 180

# TODO: filter by regulation set once I figure out how to label events


def parse_filename(filename):
    stem, ext = os.path.splitext(filename)
    if ext != '.json' or '_' not in stem:
        return None, None
    tournament_id, division = stem.split('_', 1)
    return tournament_id, division


def in_gen9_window(tournament_id):
    try:
        return GEN9_ID_MIN <= int(tournament_id) <= GEN9_ID_MAX
    except (TypeError, ValueError):
        return False


def has_decklist(player):
    decklist = player.get('decklist')
    return bool(decklist)


def opponents_for(player, by_name):
    opponents = []
    rounds = player.get('rounds') or {}

    for round_info in rounds.values():
        name = (round_info or {}).get('name')
        if not name or name.upper() == 'BYE':
            continue

        opponent = {'name': name, 'result': round_info.get('result')}
        other = by_name.get(name)
        if other is not None and has_decklist(other):
            opponent['decklist'] = other['decklist']
        opponents.append(opponent)

    return opponents


def process():
    rows = []
    skipped_files = 0

    for filename in os.listdir(RAW_DIR):
        tournament_id, division = parse_filename(filename)
        if tournament_id is None or not in_gen9_window(tournament_id):
            skipped_files += 1
            continue

        path = os.path.join(RAW_DIR, filename)
        with open(path, 'r', encoding='utf-8') as f:
            players = json.load(f)

        if not isinstance(players, list):
            skipped_files += 1
            continue

        by_name = {player.get('name'): player for player in players if player.get('name')}

        for player in players:
            if not has_decklist(player):
                continue
            rows.append({
                'tournament_id': tournament_id,
                'division': division,
                'name': player.get('name'),
                'placing': player.get('placing'),
                'decklist': player['decklist'],
                'opponents': opponents_for(player, by_name),
            })

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2)

    return rows, skipped_files


if __name__ == '__main__':
    rows, skipped_files = process()
    print(f'wrote {len(rows)} players to {os.path.abspath(OUT_PATH)}')
    print(f'skipped {skipped_files} files outside gen 9 id window or unreadable')
