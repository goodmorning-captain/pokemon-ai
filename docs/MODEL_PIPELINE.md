# Model pipeline

Path from processed PokeData standings to a learned VGC team space. This pass builds **vocab + team tensors + a match table**. Contrastive training, the generator, and the rules engine come later.

```
CANONICAL DATASET
         │
┌────────┴────────┐
↓                 ↓
Metadata          Team composition
placement         species / item / ability
division          tera / moves
tournament        (EV/IV/nature: later, Smogon)
format/reg        (regulation tag: later)
         │
         └────────┬────────┘
                  ↓
           vocab + tensors
           teams.pt
           matches.jsonl
                  ↓
           PyTorch Dataset     ← you are here
           + team encoder
           + InfoNCE loop
                  ↓
           categorical IDs
                  ↓
         ┌──────────────────┐
         │ Embedding layers │
         └────────┬─────────┘
                  ↓
           Pokémon encoder
                  ↓
             Team encoder
                  ↓
          Contrastive loss
               InfoNCE
                  ↓
         learned team space
                  ↓
            Team generator
                  ↓
            Rules engine
                  ↓
            Team scorer
                  ↓
          6-Pokémon team
```

Learned synergy and explicit constraints (type/role redundancy, speed, legality) combine into the team score. Hard constraints first; learned score second.

## Data facts

- **Species key is PokeData `id`.** It is unique per form. Always `str(id)` (`898` and `"898"` are the same). Vocab stores `id -> name` where `name` includes the form (`Urshifu [Single Strike Style]`). That name-with-form is the species token, keyed by id. Tensors store a dense integer for the id; decode with `species_id_to_name`.
- **Moves** live in `badges` (up to 4). Pad/truncate to 4.
- **Teams** may have fewer than 6 Pokémon. Pad to 6 with `PAD`. Skip empty decklists.
- **Slot order in JSON is not meaningful.** Sort the 6 slots by species id (numeric when the id is all digits) so the representation is order-invariant.
- **Tournaments do not store EVs, IVs, or natures.** The model is “show team,” not a full set. Pull Smogon stats later; do not block vocab/tensors on it.
- **`gen9.json` is the wrong training format.** Opponent `decklist`s are nested copies of other player rows. Flatten to one team per player `decklist`. Do not tensorize the nested tree as-is or the same team is counted many times.
- **Gen 9 ID window** (`20`–`180`) is already applied in `scraper/process_raw.py`. That window still mixes regulation sets.

## Artifacts (this pass)

| path | what |
|---|---|
| `data/processed/gen9.json` | nested standings (input, gitignored) |
| `data/processed/vocab.json` | `species_id_to_name` plus stoi maps; `PAD=0`, `UNK=1` |
| `data/processed/teams.pt` | integer tensors, one row per player team |
| `data/processed/matches.jsonl` | `(team_idx_a, team_idx_b, result)` when the opponent has a decklist |

Build:

```bash
python dataset/build_tensors.py
```

Primary teams are player `decklist`s. Opponent lists are used only to (1) optionally complete vocab and (2) emit match rows via `(tournament_id, division, name)` lookup—not as extra unique teams.

Each real game usually appears twice (once from each player’s perspective) if both have decklists. `result` is from `team_idx_a`’s point of view (`W`/`L`, plus rare `T` or null if the standings stored that). If `gen9.json` fails to parse, re-run `python scraper/process_raw.py`—a truncated dump will not load.

### Tensor shapes

| field | shape | notes |
|---|---|---|
| `species` | `[N, 6]` | dense id index; pad with `0` |
| `item`, `ability`, `tera` | `[N, 6]` | same |
| `moves` | `[N, 6, 4]` | from `badges` |
| `placing` | `[N]` | `-1` if missing |
| `division_id` | `[N]` | from division stoi |
| `tournament_id` | list of `N` strings | keep as-is for splits |
| `player_name` | list of `N` strings | metadata only |

## Target architecture

1. **Canonical row** = metadata + team composition, not the nested JSON object.
2. **Vocab + PAD/UNK** so every categorical field is a dense `int`. Unknown tokens at train/serve time map to `UNK`.
3. **PyTorch Dataset** later: yield the tensors above (and, for InfoNCE, two views of a team). Do not load `gen9.json` in the Dataset.
4. **Embeddings** per field → concat → small MLP → Pokémon vector.
5. **Team encoder** over 6 slots, order-invariant (DeepSets or attention). Do not feed `placing` in as an input or the generator will copy “good placing.” Optional: auxiliary loss that *predicts* top-cut.
6. **InfoNCE** on a team space. Define positives before writing the encoder (see below).
7. **Generator** samples in embedding space or slot-by-slot (species → item → moves) and **rejects** via the rules engine.
8. **Scorer** = `sim(candidate, prototype) + constraint penalties`.

## Steps still ahead (do not skip)

### Train / val split by `tournament_id`

Random row splits leak: the same team shows up against many opponents, and both sides of a match would be in train and val. Put whole tournaments in val/test. The match table joins to `teams.pt` on `team_idx_*` → `tournament_id`.

### Positive-pair definition (InfoNCE)

Need an explicit positive. Start with **augmentation** (SimCLR-style): same team, shuffle is a no-op after id-sort so instead drop one Pokémon, mask an item, or mask a move. That teaches “this is one team.”

**Do not** treat any two Masters teams as positives. **Do not** use W/L as the first contrastive signal. The match table is for a later ranking / matchup head (winner vs loser with a margin, or two teams that both beat a third).

### Regulation tagging (later)

The gen 9 id window mixes formats (e.g. restricted Miraidon + Lunala + Calyrex vs earlier SV regs). Untagged mixed regs smear archetypes in the same embedding space. Heuristic tags (restricted count, paradox presence, item clause clashes) are enough to hold out or condition on; a full official mapping can wait.

### Smogon EV / IV / nature (later)

Not in standings. When needed, join Smogon usage or sample sets onto species (+ form) as extra features or as a post-hoc set filler. Out of scope until the show-team space is useful.

### Baseline: co-occurrence

Before a net: species–species co-occurrence, item-on-species, ability-on-species. If nearest-neighbor on bag-of-species already retrieves archetypes, that is the bar InfoNCE has to beat.

### Eval

Loss is not enough.

- **kNN retrieval** in team space: neighbors of a Miraidon + Farigiraf + Hands team should look like that archetype.
- **Legality** of anything the generator emits (clause, duplicate item, illegal tera, later regulation).
- Optional: top-cut ranking if using the auxiliary head.

Until retrieval looks like archetypes, do not build the generator.

### InfoNCE Dataset / training loop

`VgcTeamDataset` (`dataset/vgc_dataset.py`) loads `teams.pt`, splits by `tournament_id`, and yields two augmented views (drop a slot, mask an item, or mask a move). `model/encoder.py` embeds each Pokémon, mean-pools the team, and projects to `z` for NT-Xent InfoNCE. Train with:

```bash
python model/train.py
```

Checkpoints go to `checkpoints/best.pt` (gitignored). `h` (pre-projector) is what later kNN retrieval should use. Matchup ranking is still a later head on `matches.jsonl`.

## Suggested order of work

1. Inspect `vocab.json` sizes and a few decoded teams from `teams.pt`.
2. Co-occurrence baseline + a cheap tournament split.
3. `Dataset` + Pokémon/team encoder + augmentation InfoNCE.
4. Regulation tags before a serious training run.
5. Matchup head on `matches.jsonl`.
6. Generator + rules engine + scorer.
7. Smogon sets if the show-team model needs real EV/IV fills.

## Assumptions

- Opponent name + `tournament_id` + `division` uniquely identifies the other player row. Matches whose name does not resolve are dropped.
- If the same PokeData id appears with two names, the first name seen (player decklists first) is kept.
- `data/processed/` is gitignored; do not commit `gen9.json`, `vocab.json`, `teams.pt`, or `matches.jsonl`.