# pokemon_ai

VGC team-modeling from PokeData standings.

```bash
python scraper/pokedata_downloader.py
python scraper/process_raw.py
python dataset/build_tensors.py
```

See [docs/MODEL_PIPELINE.md](docs/MODEL_PIPELINE.md) for the path from processed standings to vocab, tensors, and the later contrastive team space.