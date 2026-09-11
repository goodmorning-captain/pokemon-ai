import json
import os
import time

import requests
from discover_tournaments import discover_tournaments

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
REQUEST_DELAY_SECONDS = 0.5


def json_url(tournament, division):
    return f"{division['url']}{tournament['id']}_{division['name']}.json"


def download_tournaments(tournaments):
    os.makedirs(DATA_DIR, exist_ok=True)
    session = requests.Session()
    saved = []

    for tournament_name, tournament in tournaments.items():
        for division in tournament['divisions']:
            url = json_url(tournament, division)
            filename = f"{tournament['id']}_{division['name']}.json"
            out_path = os.path.join(DATA_DIR, filename)

            try:
                response = session.get(url, timeout=30)
                response.raise_for_status()
                payload = response.json()
            except requests.RequestException as exc:
                print(f'skip {url} ({exc})')
                continue

            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2)

            saved.append(out_path)
            print(f'saved {out_path} ({len(payload)} records) [{tournament_name}]')
            time.sleep(REQUEST_DELAY_SECONDS)

    return saved


if __name__ == '__main__':
    tournaments = discover_tournaments()
    print(f'discovered {len(tournaments)} tournaments')
    paths = download_tournaments(tournaments)
    print(f'downloaded {len(paths)} json files into {os.path.abspath(DATA_DIR)}')
