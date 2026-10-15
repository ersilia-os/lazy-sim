# Featurizers

How each lazy-sim featurizer represents molecules, where it comes from, how it was checked against the original code, and how well it searches.

| `-f` | Captures | Similarity | Engine | Runs on |
|---|---|---|---|---|
| `clamp` | Bioactivity-aware embedding (768 numbers) | Cosine | [turbovec](https://pypi.org/project/turbovec/), 4-bit (approximate) | RDKit + numpy |
| `ecfp` | Substructures, ECFP6 (2,048 bits) | Tanimoto | [FPSim2](https://github.com/chembl/FPSim2) (exact) | RDKit |
| `sand` | 3D shape (512 numbers) | Cosine | turbovec, 4-bit (approximate) | RDKit + numpy |

Principles shared by all featurizers:
- **SMILES are used exactly as given.** lazy-sim never standardises or strips salts; curation happens upstream.
- **No torch.** Neural featurizers are re-implemented in numpy and checked against their original code. Their weights are converted once, hosted by Ersilia, and downloaded once to `~/lazysim/models/`.
- **Two ways to search.** Every library is searched through its index. With `--embeddings`, a library also keeps every molecule's full vector, which gives the exact answer the evaluation scripts compare against.

**How the numbers below were measured.**
- **Library and queries:** the Ersilia Reference Library (1,355,109 compounds), with 1,000 of its own molecules drawn at random (seed 0) as queries. Each query's own hit is left out.
- **Machine:** a 16-thread laptop.
- **Recall:** [`scripts/evaluate_recall.py`](../scripts/evaluate_recall.py). Recall@k is the share of the indexed top k that are true top-k neighbours, where a hit counts when its exact similarity reaches that of the true k-th neighbour.
- **Time:** [`scripts/evaluate_speed.py`](../scripts/evaluate_speed.py).
  - **Indexed** is the real `lazysim search` command, from launch to exit, for 100 hits per query.
  - **Embeddings** does the same work by comparing each query with every stored embedding.
  - Files are read from disk in every run, and the median of 5 runs is shown.

```bash
python scripts/evaluate_recall.py ersilia-reference -f clamp -k 10,100,1000
python scripts/evaluate_speed.py ersilia-reference -f clamp -r 5
```

## CLAMP

### Source

> Seidl, P., Vall, A., Hochreiter, S. & Klambauer, G. Enhancing activity prediction models in drug discovery with the ability to understand human language. *ICML* (2023).

- **Code:**
  - [ml-jku/clamp](https://github.com/ml-jku/clamp) at commit `61575cb`.
  - Its fingerprints come from [mhnreact](https://github.com/ml-jku/mhn-react) at commit `a3e0cc6`.
- **Licence:**
  - The software is GPL-3.0.
  - The model terms say the model and its derivatives must not be used to exploit or harm, or to attempt to exploit or harm, in any way.

### What lazy-sim computes

**Fingerprint.** It reproduces mhnreact's `getFingerprint(which="morganc+rdkc", fp_size=8192)`, quirks included:
1. Take a Morgan count fingerprint (radius 2, feature invariants, chirality, bond types) and an RDKit path count fingerprint (paths up to 6 bonds), both unfolded.
2. Fold each into 8,192 slots by `key mod 8192`. When two keys of the same fingerprint land in one slot, the last one wins rather than being added; this is mhnreact's behaviour.
3. Add the two folded fingerprints and apply `log1p`.

**Encoder.** It's CLAMP's compound encoder, an MLP of 8,192 → 4,096 → 2,048 → 768.
- Each of the two hidden layers is a linear layer, then LayerNorm without scale or shift, then ReLU.
- The last layer is linear.
- The output is L2-normalised, so cosine similarity is a dot product.

The code is in [`src/lazy_sim/featurizers/clamp.py`](../src/lazy_sim/featurizers/clamp.py).

### Weights

**Original:** the pretrained JKU checkpoint (667 MB, sha256 `fc41395…`). Only the compound encoder is kept.

**Conversion:** [`scripts/export_clamp_weights.py`](../scripts/export_clamp_weights.py), which needs torch and the original CLAMP code and is for maintainers only.

**The converted file:** `clamp_compound_encoder_v1.npz`.

| Array | Shape |
|---|---|
| `w1` | 8192×4096 |
| `w2` | 4096×2048 |
| `w3` | 2048×768 |

- It also holds the matching biases.
- Size: 174,092,702 bytes.
- sha256: `9ddd688c60af683fb1c70cc0610464e12446c05f9d04869420f338c0e80ee18b`.

**Hosting:** on Ersilia's public S3 at [`lazy-sim/data/clamp/`](https://eosvc-public.s3.eu-central-1.amazonaws.com/lazy-sim/data/clamp/clamp_compound_encoder_v1.npz).

**Download and checks:**
- lazy-sim downloads the file once, the first time a command needs it. `LAZYSIM_CLAMP_WEIGHTS` points to a local copy instead.
- It checks the sha256 at every load.
- Each library records the sha256 of the weights it was built with, and `search` refuses a library built with other weights.

### Checked against the original

- **During development:**
  - Fingerprints were identical to mhnreact's on 10,000 ChEMBL molecules.
  - Embeddings were within 6e-7 of the original `encode_smiles`.
- **Kept as a test:** `tests/test_cli.py::test_clamp_matches_original`.
  - It embeds 20 molecules and compares them with embeddings from the original code (`tests/data/clamp_reference.npy`).
  - The largest allowed difference is 1e-5.
- **Known differences:** none.

### Search quality and speed

| | Ersilia Reference Library |
|---|---|
| Index (turbovec, 4-bit) | 527 MB (389 B per molecule) |
| Embeddings (float32) | 4.16 GB (3 KB per molecule) |
| Recall@10 / @100 / @1000 | 0.955 / 0.952 / 0.953 |
| Score error (mean) | 0.0018 |
| Time indexed | 7.5 s |
| Time embeddings | 15.2 s |

- **ChEMBL 37:** on 2.9M compounds, with 1,000 queries held out of the library, recall@100 was 0.958.
- **Scores are compressed:** random pairs score about 0.5 and close analogues above 0.9. CLAMP was trained on bioactivity data, so its cosine is not a Tanimoto coefficient.

## ECFP

### What lazy-sim computes

- **Fingerprint:** ECFP6, RDKit's Morgan fingerprint of radius 3 with 2,048 bits. It uses the RDKit generator's defaults: bits rather than counts, and no chirality.
- **Similarity:** Tanimoto.
- **Code:** [`src/lazy_sim/featurizers/ecfp.py`](../src/lazy_sim/featurizers/ecfp.py).

### Engine

- **FPSim2:** [FPSim2](https://github.com/chembl/FPSim2) 0.7.4 (MIT), installed with `pip install "lazy-sim[ecfp]"`.
- **Index:** FPSim2 builds `ecfp.h5` with the same fingerprint parameters.
- **Search:** exact Tanimoto top-k, with one query per thread.
- **Embeddings:** `--embeddings` stores the fingerprints as packed bits (`np.unpackbits` gives the 2,048 bits back).

### Checked

- **Test:** `tests/test_cli.py::test_search_ecfp` checks that FPSim2's scores equal the Tanimoto computed from the stored bits.
- **Against an independent search:** `evaluate_recall.py` compares FPSim2 with a separate numpy Tanimoto over the stored bits.
  - Recall was 1.000 at every k.
  - The score error was 0.

### Search quality and speed

| | Ersilia Reference Library |
|---|---|
| Index (`ecfp.h5`) | 245 MB |
| Embeddings (packed bits) | 347 MB (256 B per molecule) |
| Recall@10 / @100 / @1000 | 1.000 / 1.000 / 1.000 |
| Score error (mean) | 0.0000 |
| Time indexed | 10.1 s |
| Time embeddings | 26.6 s |

## SAND

### Source

- **Authors:** Winter et al. (Pfizer).
  - Preprint on [OpenReview](https://openreview.net/forum?id=pB6WAdnRDR) (2026).
  - Code at [pfizer-opensource/SAND](https://github.com/pfizer-opensource/SAND), Apache-2.0.
- **Weights:** the `sand_no_comp.ckpt` checkpoint from release v1.0.0, which gives the uncompressed embedding.
- **On the Ersilia Model Hub:** [eos5mnx](https://github.com/ersilia-os/eos5mnx).
- **What it captures:** the cosine between SAND embeddings approximates ROCS 3D shape overlap (correlation 0.86, per the authors), without generating conformers.

### What lazy-sim computes

**Graph.** It reproduces SAND's `RDKitGraphFeaturizer`, built with RDKit from the SMILES, with no hydrogens added and no 3D coordinates.
- RDKit's legacy stereo perception (`AssignStereochemistry`) labels the stereocentres first.
- Each atom is 16 categories: element, degree, formal charge, hydrogens, aromaticity, hybridisation, ring membership and ring sizes 3–10, and CIP label.
- Each bond is 5 categories: order, ring, aromaticity, rotatable, and E/Z.
- Every bond is used in both directions.

**Network.** 27.9M parameters, run in numpy.
1. Summed category embeddings (1,024 numbers per atom and per bond).
2. Seven GINE message-passing layers. Each has a 3-layer SiLU MLP, a LayerNorm over each molecule's atoms and channels together, and a residual connection.
3. Softmax-attention pooling to one vector per molecule.
4. An MLP of 1,024 → 1,024 → 1,024 → 512 with LayerNorm.
5. L2 normalisation.

**Differences from the original.** Molecules with 255 atoms or more crash the original. lazy-sim embeds them, so every library row gets a vector.

The code is in [`src/lazy_sim/featurizers/sand.py`](../src/lazy_sim/featurizers/sand.py).

### Weights

**Original:** `sand_no_comp.ckpt` from SAND release v1.0.0 (335,091,149 bytes, sha256 `dd022d49…`). Only the encoder is kept.

**Conversion:** [`scripts/export_sand_weights.py`](../scripts/export_sand_weights.py), which needs only torch and is for maintainers. With SAND's own code, it also writes the reference embeddings for the tests.

**The converted file:** `sand_encoder_v1.npz`.
- It holds 83 arrays under SAND's own parameter names.
- Size: 111,687,496 bytes.
- sha256: `ee72a20915dc2ba44cbce0812fa34868416d38aeebb88ffb379f38a2f10d4501`.
- SAND's Apache-2.0 licence is kept next to it.

**Hosting:** on Ersilia's public S3 at `lazy-sim/data/sand/`.

**Download and checks:** the same as CLAMP. It is downloaded once to `~/lazysim/models/` (`LAZYSIM_SAND_WEIGHTS` points to a local copy), and its sha256 is checked at every load and recorded in each library.

### Checked against the original

- **The reference:** 10,000 random molecules of the Ersilia Reference Library, embedded with SAND's own code.
  - The code was the copy in eos5mnx, commit `711499a`.
  - It ran with torch 2.4.1 and RDKit 2024.3.5.
- **Result:**
  - 9,998 molecules agree within 7e-7, with a median difference of 1e-7.
  - Two differ (cosine 0.88 and 0.96 to the original). Both come from RDKit's version: lazy-sim uses RDKit 2026.3.6, which perceives these two molecules differently.
    - In one, a furan ring inside a large ring system is no longer aromatic.
    - In the other, a sulfur is sp3 rather than sp2.
  - Given the same graph, the port reproduces the original.
- **Kept as a test:** `tests/test_cli.py::test_sand_matches_original`.
  - It embeds 20 molecules and compares them with embeddings from SAND's own code (`tests/data/sand_reference.npy`).
  - The largest allowed difference is 1e-5.

### Speed

SAND does about 50× more work per molecule than CLAMP. On a 16-thread laptop:

| | Molecules per second |
|---|---|
| lazy-sim (numpy, one process) | ~55–60 |
| Original (torch, 16 threads) | 120 |

- **The setting that helps:** lazy-sim lets OpenBLAS threads sleep as soon as each matrix product ends (`OPENBLAS_THREAD_TIMEOUT=4`, unless you set it yourself).
  - Otherwise they spin for about 0.1 s and slow the single-core numpy steps in between.
  - It makes SAND about 20% faster (48 → 59 molecules per second) and CLAMP indexing about 6% faster, and doesn't change CLAMP search.
- **Queries:** embedding 1,000 queries takes about 17–20 s.
- **Indexing:** the Ersilia Reference Library took 6 h 21 min (about 59 molecules per second, 3.4 GB of memory at peak). That is paid once, by Ersilia, for the libraries `fetch` serves.

### Search quality and speed

| | Ersilia Reference Library |
|---|---|
| Index (turbovec, 4-bit) | 353 MB (260 B per molecule) |
| Embeddings (float32) | 2.78 GB (2 KB per molecule) |
| Recall@10 / @100 / @1000 | 0.967 / 0.966 / 0.959 |
| Score error (mean) | 0.0033 |
| Time indexed | 24.7 s |
| Time embeddings | 30.0 s |

- **Where the time goes:** embedding the 1,000 queries takes about 20 s, so SAND's indexed search is only a little faster than going through every embedding. The search itself takes a few seconds.
