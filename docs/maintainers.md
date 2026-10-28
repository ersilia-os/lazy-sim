# Maintainer guide

How Ersilia develops lazy-sim and builds, evaluates and publishes the libraries that `lazysim fetch` serves. Uploads to S3 need Ersilia's AWS credentials (`eosvc config`).

## Setup and tests

```bash
pip install -e ".[ecfp,dev]"
ruff check . && ruff format .
LAZYSIM_CLAMP_WEIGHTS=data/clamp/clamp_compound_encoder_v1.npz \
LAZYSIM_SAND_WEIGHTS=data/sand/sand_encoder_v1.npz \
LAZYSIM_MOLJEPA_WEIGHTS=data/moljepa/moljepa_4c912b4.safetensors \
pytest
```

- Without the `LAZYSIM_<NAME>_WEIGHTS` variables, the tests download the weights into a temporary home.
- The suite takes about 2 minutes. It runs offline, with `fetch` reading a catalog of `file://` URLs.
- `tests/data/` holds a small library, queries, and each neural featurizer's embeddings from its original code (`*_reference.npy`).

## Data and eosvc

`data/` is gitignored. [eosvc](https://github.com/ersilia-os/eosvc) backs it with the public S3 bucket `eosvc-public`, under `lazy-sim/data/`.

| Folder | Holds |
|---|---|
| `data/clamp/`, `data/sand/` | Converted weights and the original licences |
| `data/moljepa/` | A local copy of Mol-JEPA's weights, not uploaded |
| `data/libraries/` | The published libraries, one folder each, and `catalog.json` |
| `data/logs/` | Build and evaluation logs, local only |

`eosvc download --path data/clamp` gets a folder. `eosvc upload --path <file or folder>` sends one.

## Featurizer weights

- `scripts/export_clamp_weights.py` and `export_sand_weights.py` convert the original checkpoint to an npz file once. They need torch, which lazy-sim doesn't depend on.
- With the original code and a SMILES file, the same scripts also write the test reference embeddings. Each script's docstring lists the environment it needs.
- Mol-JEPA's weights are read from Hugging Face at a pinned revision, so `export_moljepa_reference.py` writes only the reference embeddings.
- Each featurizer module pins its weights' URL and sha256 (`WEIGHTS_URL`, `WEIGHTS_SHA256`). New weights need a new file name, new constants and a re-index of every library.
- Upload with `eosvc upload --path data/<featurizer>`.

## Building a library

`scripts/build_library.py` builds the libraries Ersilia maintains: the Ersilia Reference Library and the libraries of the [Ersilia Maintained Inputs](https://github.com/ersilia-os/ersilia-model-hub-maintained-inputs). `--list` shows them with their size.

```bash
python scripts/build_library.py --list
python scripts/build_library.py <library> -t 10                      # every featurizer
python scripts/build_library.py <library> -f ecfp                    # one featurizer
python scripts/build_library.py <library> --add <featurizer> -t 10   # add one to a published library
```

- **Names** follow the Ersilia Maintained Inputs: `<library>_<size>_v<N>`, such as `ersilia_reference_library_1M_v0`.
- **Sources.** The Reference Library is pinned to a commit and a sha256. The others are files in the public S3 bucket. The script checks each source's number of molecules and its sha256, and prints the sha256 of a source that isn't pinned yet for you to pin in `LIBRARIES`. A repeated id or SMILES stops the build, since a library should have none.
- **Invalid SMILES** are left out of the index and listed in the library's `skipped.csv`. The registry's `molecules` is the rows of the source, and the catalog's is the molecules indexed, so they differ by the skipped ones (COCONUT: 715,822 and 715,818).
- **`catalog.json`** is the one on S3 plus the libraries in `data/libraries/` that aren't uploaded yet, so building one library never drops another. For a library on both, S3's entry wins.
- **Resuming.** Each featurizer is built on its own, fastest first (ECFP, CLAMP, Mol-JEPA, SAND), and kept in `tmp/build/<library>/index`. After a stop, run the same command to build only the missing ones; an interrupted featurizer starts again from its beginning. The library is published once all four exist, and the seconds each one took are in `tmp/build/<library>/timings.json`.
- **Embeddings** are kept by default, for users who `fetch` them, and take about 7 GB per million molecules over the four featurizers. A featurizer is indexed once, so `--no-embeddings` cannot be undone later.
- **Published files are never edited.** A rebuild is published under the next version, such as `_v1`. `--add` is the one exception. It only adds the featurizer's files and catalog entry, so rows stay aligned for everyone who already fetched the library.
- **Restrictive licences** go in the script's `LICENSES`. They are stored in the catalog, and `lazysim catalog --remote` prints them.
- **Long builds** run detached, with memory and CPU caps. numpy is loaded before `-t` is read, so set the thread variables too. Follow progress with `journalctl --user -u lazysim-build -f`.

  ```bash
  systemd-run --user --unit=lazysim-build -p MemoryMax=8G -p MemorySwapMax=0 \
    -p CPUQuota=1000% --nice=19 --working-directory=$PWD \
    --setenv=LAZYSIM_SAND_WEIGHTS=$PWD/data/sand/sand_encoder_v1.npz \
    --setenv=OPENBLAS_NUM_THREADS=10 --setenv=OMP_NUM_THREADS=10 \
    --setenv=MKL_NUM_THREADS=10 --setenv=RAYON_NUM_THREADS=10 \
    $(which python) scripts/build_library.py <library> -t 10
  ```

  On a 16-thread laptop, SAND took 6 h 21 min and Mol-JEPA 5 h 7 min for the Reference Library. With 14 threads, the Enamine Hit Locator Library (460K) took 1 h 49 min and 1 h 27 min, and COCONUT (715K), whose molecules are larger, 4 h 19 min and 3 h 43 min. ECFP and CLAMP take minutes.

## Evaluating

```bash
python scripts/evaluate_recall.py data/libraries/<name> -f <featurizer> -k 10,100,1000
python scripts/evaluate_speed.py data/libraries/<name> -f <featurizer> -r 5
```

- Both need the featurizer's embeddings.
- Run `evaluate_speed.py` on an idle machine, with all threads.
- The results go in the featurizer's section of [featurizers.md](featurizers.md) and in its table at the top.

## Comparing featurizers

```bash
python scripts/compare_featurizers.py <name> -f <a> -f <b> -t 6 --save
```

- This writes the pair into `src/lazy_sim/results/<name>.json`, which `lazysim compare` reads. Commit that file.
- Each pair takes 2–3 minutes and 4–8 GB of memory on the Reference Library.
- Run the pairs one after another, since they all write to the same file.
- Then update "All pairs" in [featurizers.md](featurizers.md#all-pairs).

## Publishing

S3 layout, under `lazy-sim/data/libraries/`:
- `catalog.json` lists every library and featurizer, with each file's path, size and sha256.
- `<name>/` holds `library.db`, `manifest.json` and `skipped.csv`. Each featurizer adds `<featurizer>.json`, its index (`.tv` or `.h5`) and `<featurizer>_embeddings.npy`.

1. Upload the library's new files: `eosvc upload --path data/libraries/<name>/<file>`.
2. Upload `data/libraries/catalog.json` last, so the catalog never lists a missing file.
3. Check from an empty home: `LAZYSIM_HOME=$(mktemp -d) lazysim fetch <name> -f <featurizer>`, then `lazysim search` on a few queries.
4. Update the Libraries table in the README.

`LAZYSIM_CATALOG_URL` points lazy-sim at another catalog, for example a local `file://` one.

## Releases

lazy-sim hasn't been released yet. Versions follow semantic versioning (`vMAJOR.MINOR.PATCH`). PyPI releases will come from a GitHub Action that runs on a GitHub release. The tag, the release name and `version` in `pyproject.toml` must match.
