# Featurizers

How each lazy-sim featurizer represents molecules, where it comes from, how it was checked against the original code, how well it searches, and [how the featurizers relate](#how-the-featurizers-relate).

| `-f` | Captures | Similarity, engine | Licence | Ersilia Model Hub |
|---|---|---|---|---|
| `clamp` | Bioactivity-aware embedding (768 numbers) | Cosine, [turbovec](https://pypi.org/project/turbovec/) 4-bit (approximate) | GPL-3.0, plus the model terms | [eos3l5f](https://github.com/ersilia-os/eos3l5f) |
| `ecfp` | Substructures, ECFP6 (2,048 bits) | Tanimoto, [FPSim2](https://github.com/chembl/FPSim2) (exact) | No model | [eos4wt0](https://github.com/ersilia-os/eos4wt0) |
| `sand` | 3D shape (512 numbers) | Cosine, turbovec 4-bit (approximate) | Apache-2.0 | [eos5mnx](https://github.com/ersilia-os/eos5mnx) |
| `moljepa` | Biological, phenotypic and quantum context predicted from the graph (512 numbers) | Cosine, turbovec 4-bit (approximate) | **CC BY-NC 4.0, non-commercial use only** | [eos9q2i](https://github.com/ersilia-os/eos9q2i) |

SAND and Mol-JEPA were checked against the code of their Hub models. CLAMP was checked against the original ml-jku code, not against eos3l5f. ECFP hasn't been compared with eos4wt0 yet.

On the Ersilia Reference Library, measured as described in [How the numbers were measured](#how-the-numbers-were-measured):

| `-f` | Recall@100 | 1,000 queries, indexed | 1,000 queries, embeddings | Index | Embeddings | A random molecule scores |
|---|---|---|---|---|---|---|
| [`clamp`](#clamp) | 0.952 | 7.5 s | 15.2 s | 527 MB | 4.16 GB | 0.52 |
| [`ecfp`](#ecfp) | 1.000 | 10.1 s | 26.6 s | 245 MB | 347 MB | 0.09 |
| [`sand`](#sand) | 0.966 | 24.7 s | 30.0 s | 353 MB | 2.78 GB | 0.06 |
| [`moljepa`](#mol-jepa) | 0.963 | 18.0 s | 23.1 s | 353 MB | 2.78 GB | 0.12 |

The last column is the typical similarity of a random library molecule to a query. Scores are on different scales, so a score only means something next to its featurizer's own baseline.

Recall@100 on the other libraries, measured the same way (sizes and times are in each featurizer's section):

| `-f` | Reference Library (1,355,109) | Enamine Hit Locator (460,160) | COCONUT (715,818) |
|---|---|---|---|
| `clamp` | 0.952 | 0.952 | 0.916 |
| `ecfp` | 1.000 | 1.000 | 1.000 |
| `sand` | 0.966 | 0.972 | 0.935 |
| `moljepa` | 0.963 | 0.970 | 0.902 |

COCONUT is the hardest library for the 4-bit indexes. Its recall@10 is 0.876 for CLAMP, 0.924 for SAND and 0.852 for Mol-JEPA, and it rises with k. Why hasn't been investigated. ECFP is exact on every library.

Principles shared by all featurizers:
- **SMILES are used exactly as given.** lazy-sim never standardises or strips salts. Curation happens upstream.
- **No torch.** Everything runs on RDKit and numpy, plus FPSim2 for ECFP. Neural featurizers are re-implemented in numpy and checked against their original code. Their weights are downloaded once to `~/lazysim/models/`: converted and hosted by Ersilia, except Mol-JEPA's, which are read as released on Hugging Face.
- **Two ways to search.** Every library is searched through its index. With `--embeddings`, a library also keeps every molecule's full vector, which gives the exact answer the evaluation scripts compare against.

## How the numbers were measured

- **Library and queries:** the Ersilia Reference Library (1,355,109 compounds), with 1,000 of its own molecules drawn at random (seed 0) as queries. Each query's own hit is left out. The Enamine Hit Locator Library (460,160) and COCONUT (715,818) were measured the same way.
- **Machine:** a 16-thread laptop for the Reference Library, and a 16-thread Intel Core i5-1340P for the Enamine Hit Locator Library and COCONUT, so times are not strictly comparable between the two.
- **Recall:** [`scripts/evaluate_recall.py`](../scripts/evaluate_recall.py). Recall@k is the share of the indexed top k that are true top-k neighbours, where a hit counts when its exact similarity reaches that of the true k-th neighbour.
- **Time:** [`scripts/evaluate_speed.py`](../scripts/evaluate_speed.py).
  - **Indexed** is the real `lazysim search` command, from launch to exit, for 100 hits per query.
  - **Embeddings** does the same work by comparing each query with every stored embedding.
  - Files are read from disk in every run, and the median of 5 runs is shown.
- **How two featurizers relate:** [`scripts/compare_featurizers.py`](../scripts/compare_featurizers.py), on the hits `lazysim search` returns and on exact rankings. See [the last section](#how-the-featurizers-relate).

```bash
python scripts/evaluate_recall.py ersilia_reference_library_1M_v0 -f clamp -k 10,100,1000
python scripts/evaluate_speed.py ersilia_reference_library_1M_v0 -f clamp -r 5
python scripts/compare_featurizers.py ersilia_reference_library_1M_v0 -f clamp -f ecfp
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
- **On the Ersilia Model Hub:** [eos3l5f](https://github.com/ersilia-os/eos3l5f). lazy-sim's port was checked against the original code above, not against eos3l5f.

### What lazy-sim computes

**Fingerprint.** It reproduces mhnreact's `getFingerprint(which="morganc+rdkc", fp_size=8192)`, quirks included:
1. Take a Morgan count fingerprint (radius 2, feature invariants, chirality, bond types) and an RDKit path count fingerprint (paths up to 6 bonds), both unfolded.
2. Fold each into 8,192 slots by `key mod 8192`. When two keys of the same fingerprint land in one slot, the last one wins rather than being added. This is mhnreact's behaviour.
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

| | Ersilia Reference Library | Enamine Hit Locator | COCONUT |
|---|---|---|---|
| Index (turbovec, 4-bit) | 527 MB (389 B per molecule) | 179 MB | 279 MB |
| Embeddings (float32) | 4.16 GB (3 KB per molecule) | 1.41 GB | 2.20 GB |
| Recall@10 / @100 / @1000 | 0.955 / 0.952 / 0.953 | 0.950 / 0.952 / 0.954 | 0.876 / 0.916 / 0.946 |
| Score error (mean) | 0.0018 | 0.0015 | 0.0015 |
| Time indexed | 7.5 s | 3.4 s | 4.7 s |
| Time embeddings | 15.2 s | 5.9 s | 8.1 s |

- **ChEMBL 37:** on 2.9M compounds, with 1,000 queries held out of the library, recall@100 was 0.958.
- **Scores are compressed:** random pairs score about 0.5 and close analogues above 0.9. CLAMP was trained on bioactivity data, so its cosine is not a Tanimoto coefficient.

## ECFP

### What lazy-sim computes

- **Fingerprint:** ECFP6, RDKit's Morgan fingerprint of radius 3 with 2,048 bits. It uses the RDKit generator's defaults: bits rather than counts, and no chirality.
- **Similarity:** Tanimoto.
- **Code:** [`src/lazy_sim/featurizers/ecfp.py`](../src/lazy_sim/featurizers/ecfp.py).
- **On the Ersilia Model Hub:** [eos4wt0](https://github.com/ersilia-os/eos4wt0), Morgan fingerprints in binary form (radius 3, 2,048 bits). lazy-sim's fingerprints haven't been compared with it yet.

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

| | Ersilia Reference Library | Enamine Hit Locator | COCONUT |
|---|---|---|---|
| Index (`ecfp.h5`) | 245 MB | 81 MB | 120 MB |
| Embeddings (packed bits) | 347 MB (256 B per molecule) | 118 MB | 183 MB |
| Recall@10 / @100 / @1000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| Score error (mean) | 0.0000 | 0.0000 | 0.0000 |
| Time indexed | 10.1 s | 3.3 s | 4.6 s |
| Time embeddings | 26.6 s | 8.4 s | 13.3 s |

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
- **Indexing:** the Ersilia Reference Library took 6 h 21 min (about 59 molecules per second, 3.4 GB of memory at peak). That is paid once, by Ersilia, for the libraries `fetch` serves. On 14 threads, the Enamine Hit Locator Library took 1 h 49 min (about 70 molecules per second) and COCONUT, whose molecules are larger, 4 h 19 min (about 46 per second).

### Search quality and speed

| | Ersilia Reference Library | Enamine Hit Locator | COCONUT |
|---|---|---|---|
| Index (turbovec, 4-bit) | 353 MB (260 B per molecule) | 120 MB | 187 MB |
| Embeddings (float32) | 2.78 GB (2 KB per molecule) | 942 MB | 1.47 GB |
| Recall@10 / @100 / @1000 | 0.967 / 0.966 / 0.959 | 0.970 / 0.972 / 0.969 | 0.924 / 0.935 / 0.952 |
| Score error (mean) | 0.0033 | 0.0032 | 0.0029 |
| Time indexed | 24.7 s | 16.4 s | 25.1 s |
| Time embeddings | 30.0 s | 18.1 s | 27.9 s |

- **Where the time goes:** embedding the 1,000 queries takes about 20 s, so SAND's indexed search is only a little faster than going through every embedding. The search itself takes a few seconds.

## Mol-JEPA

### Source

- **Authors:** Boehringer Ingelheim.
  - Preprint on [arXiv](https://arxiv.org/abs/2608.22642) (2026).
  - Code at [Boehringer-Ingelheim/mol-jepa](https://github.com/Boehringer-Ingelheim/mol-jepa).
  - Weights on Hugging Face at [Flogrammer/Mol-JEPA](https://huggingface.co/Flogrammer/Mol-JEPA).
- **Licence: CC BY-NC 4.0, non-commercial use only.**
  - The licence covers the code and the weights.
  - Commercial use needs the authors' written permission.
  - lazy-sim prints this notice whenever it needs the weights, and `lazysim catalog --remote` shows it below the table for published Mol-JEPA indexes.
- **On the Ersilia Model Hub:** [eos9q2i](https://github.com/ersilia-os/eos9q2i).
- **What it captures:**
  - Mol-JEPA was trained by masking whole modalities across 4.69M compounds: a molecular graph, ECFP4, quantum-chemistry and structural embeddings, descriptors, and bioactivity labels.
  - Per the authors, its embedding carries predicted biological, phenotypic and quantum context.
  - From a SMILES, only the graph is observed. The other modalities are treated as missing.

### What lazy-sim computes

**Graph.** It reproduces molfeat 0.10.1's `AdjGraphTransformer`, with explicit hydrogens and canonical atom order.
- The molecule is parsed from the SMILES, all hydrogens become atoms, and the atoms are renumbered by RDKit's canonical ranking. Nothing else is done to the molecule.
- **82 atom features:** chirality tag, degree, formal charge, hybridisation, implicit valence, aromatic, possible chiral centre, in ring, radical electrons, element (43 symbols, H included) and H count.
- **17 bond features:** direction, conjugated, in ring, stereo, type, topological distance and shared ring.
- molfeat concatenates its feature blocks in alphabetical order, and lazy-sim keeps that order.
- Every bond is used in both directions.

**Network.** About 27M of Mol-JEPA's 45M parameters, run in numpy.
1. A linear layer, 82 → 512.
2. Three graph-attention layers (PyG `TransformerConv`).
   - Each has 8 heads, each 512 wide, averaged. Bond features are added to the keys and values.
   - Each is followed by GELU, a residual connection and LayerNorm.
3. Max pooling over atoms, then a linear layer, 512 → 512: the graph token.
4. A summary token and the graph token, each with a learned position, go through Mol-JEPA's 2-layer transformer: 4 heads, normalisation before each block, a 2,048-wide feed-forward, and a final LayerNorm. The missing modalities are masked out.
5. A linear layer on the summary token gives the 512 numbers.

**Differences from the original.** lazy-sim L2-normalises the output so cosine is a dot product. The original's vectors have lengths of about 2.2–2.6.

The code is in [`src/lazy_sim/featurizers/moljepa.py`](../src/lazy_sim/featurizers/moljepa.py).

### Weights

- **Source:** read directly from Hugging Face's `model.safetensors`, pinned to revision `4c912b4`.
  - Size: 181,651,424 bytes.
  - sha256: `a443592193075334b55483501f1e04cdf4ef9c461db103f687ba4335b036cf14`.
- **No conversion and no re-hosting.** lazy-sim reads the 92 tensors it needs, out of 216, with numpy alone.
- **Download and checks:**
  - Downloaded once to `~/lazysim/models/`. `LAZYSIM_MOLJEPA_WEIGHTS` points to a local copy instead.
  - The sha256 is checked at every load and recorded in each library.

### Checked against the original

- **The reference:** Reference Library molecules embedded with Mol-JEPA's own code, run offline from the pinned snapshot that eos9q2i stores.
  - The setup was torch 2.8, transformers 4.50.3, molfeat 0.10.1 and RDKit 2026.3.4.
- **Result on 10,000 random molecules:**
  - 9,999 graphs are identical to molfeat's, features and edge order, and their embeddings agree within 1e-6 (median 8e-8).
  - The remaining molecule differs because of RDKit's version: in a furan inside a large ring system, RDKit 2026.3.4 sees 5 aromatic atoms and 2026.3.6, which lazy-sim uses, sees none. Its cosine to the original is 0.90. SAND's check flagged the same molecule.
- **Kept as a test:** `tests/test_cli.py::test_moljepa_matches_original`.
  - It embeds 20 molecules and compares them with the original's embeddings (`tests/data/moljepa_reference.npy`).
  - Those reference embeddings come from [`scripts/export_moljepa_reference.py`](../scripts/export_moljepa_reference.py), which runs Mol-JEPA's own code offline from the pinned snapshot. It's the counterpart of the other featurizers' export scripts, with no weights to export.
  - The largest allowed difference is 1e-5.

### Speed

On a 16-thread laptop:

| | Molecules per second |
|---|---|
| lazy-sim (numpy, one process) | ~65–75 |
| Original (torch 2.8, 16 threads, including molfeat) | 28 |

- **Why the port is faster:** lazy-sim rearranges each attention layer with the same result, so the work per bond is on 512-wide inputs and 17 bond features rather than 4,096-wide keys and values.
  - The query absorbs the key and bond projections.
  - The value and bond projections are applied once per atom, after summing over its bonds.
- **Graphs are large:** with hydrogens as atoms, a Reference Library molecule averages 54 atoms.
- **Queries:** embedding 1,000 queries takes about 15 s.
- **Indexing:** the Ersilia Reference Library took 5 h 7 min (about 74 molecules per second, 3.8 GB of memory at peak), on 10 of the 16 threads. On 14 threads, the Enamine Hit Locator Library took 1 h 27 min (about 88 molecules per second) and COCONUT 3 h 43 min (about 54 per second).

### Search quality and speed

| | Ersilia Reference Library | Enamine Hit Locator | COCONUT |
|---|---|---|---|
| Index (turbovec, 4-bit) | 353 MB (260 B per molecule) | 120 MB | 187 MB |
| Embeddings (float32) | 2.78 GB (2 KB per molecule) | 942 MB | 1.47 GB |
| Recall@10 / @100 / @1000 | 0.957 / 0.963 / 0.969 | 0.966 / 0.970 / 0.974 | 0.852 / 0.902 / 0.934 |
| Score error (mean) | 0.0027 | 0.0029 | 0.0021 |
| Time indexed | 18.0 s | 13.5 s | 21.6 s |
| Time embeddings | 23.1 s | 15.4 s | 24.4 s |

- **Where the time goes:** as with SAND, embedding the 1,000 queries takes most of it (about 15 s). The search itself takes a few seconds.

## How the featurizers relate

Two featurizers can both find good neighbours and still disagree on which molecules those are. To see how far they agree, Ersilia takes 1,000 molecules of a library as queries (the same ones as above), asks both featurizers for each query's top k, and compares the two lists.

The results come with lazy-sim, so the library need not be fetched:

```bash
lazysim compare ersilia_reference_library_1M_v0 -f clamp -f ecfp
```

For the Ersilia Reference Library, every pair of CLAMP, ECFP, SAND and Mol-JEPA is stored.

### How to read the table

**Each cell is a median and a range over the 1,000 queries.** For example, `0.40 [0.20–0.60]` means the typical query scores 0.40, and the middle half of the queries score between 0.20 and 0.60 (first to third quartile).

**Three measures, one block each:**
1. **Molecules in both top-k lists:** the share of the top k that both featurizers return. 1 means the same molecules, 0 means none in common.
2. **Similarity to the query,** under one featurizer's own score (cosine, or Tanimoto for ECFP), computed exactly from the embeddings. Each block has three rows:
   - That featurizer's own top k. These are the most similar molecules by its own score, so they are the ceiling.
   - Random library molecules, the floor.
   - The other featurizer's top k. Where it falls between the two says how similar the other's picks look to this featurizer.

   Compare rows within a block, never across blocks: the scales differ. A CLAMP cosine of 0.5 is a random molecule, while an ECFP Tanimoto of 0.5 is a close analogue.
3. **Position in the other's ranking:** sort the whole library by one featurizer's similarity to the query, with 1 the most similar. The row shows where the other featurizer's top k sit in that order, as the median over the k molecules. A random molecule sits in the middle, at 677,554 of 1,355,109. A molecule tied with others in score takes the best of their positions.

**What `lazysim search` returns, or exact rankings.** `lazysim compare` shows two tables:
- The first compares the hits the indexes return (turbovec for the neural featurizers, FPSim2 for ECFP).
- The second compares exact rankings computed from the embeddings, so the indexes play no part.

In both, similarities and positions are computed exactly, because an index only scores the hits it returns.

### CLAMP vs ECFP

What `lazysim search` returns:

| | k = 10 | k = 100 | k = 1,000 |
|---|---|---|---|
| **Molecules in both top-k lists** | 0.40 [0.20–0.60] | 0.18 [0.09–0.31] | 0.08 [0.04–0.13] |
| **CLAMP cosine to the query** | | | |
| CLAMP's top k | 0.91 [0.88–0.94] | 0.85 [0.83–0.88] | 0.80 [0.78–0.83] |
| ECFP's top k | 0.86 [0.79–0.91] | 0.73 [0.68–0.78] | 0.64 [0.59–0.68] |
| Random library molecules | 0.52 [0.47–0.55] | 0.52 [0.47–0.55] | 0.52 [0.47–0.55] |
| **ECFP Tanimoto to the query** | | | |
| ECFP's top k | 0.52 [0.45–0.59] | 0.35 [0.31–0.39] | 0.25 [0.23–0.27] |
| CLAMP's top k | 0.41 [0.29–0.52] | 0.21 [0.17–0.26] | 0.14 [0.12–0.16] |
| Random library molecules | 0.09 [0.08–0.10] | 0.09 [0.08–0.10] | 0.09 [0.08–0.10] |
| **Position in the other's ranking (of 1,355,109)** | | | |
| CLAMP's top k, in ECFP's ranking | 21 [8–325] | 10,520 [831–66,692] | 158,711 [79,738–267,025] |
| ECFP's top k, in CLAMP's ranking | 23 [8–716] | 10,532 [1,031–50,368] | 151,306 [75,542–251,020] |
| A random molecule | 677,554 | 677,554 | 677,554 |

What it shows:
- **They agree on the closest analogues.** A typical query gets 4 of the same molecules in both top 10s. The typical molecule of each top 10 sits around 21st–23rd in the other's ranking, out of 1.36M.
- **They part quickly after that.** Only 18% of the top 100 and 8% of the top 1,000 are shared. The typical position of CLAMP's top 100 in ECFP's ranking is about 10,500th. That is still within the top 1% of the library, far from a random molecule's 677,554th.
- **To each featurizer, the other's picks are similar but not the most similar.** Take the top 100:
  - ECFP scores CLAMP's top 100 at 0.21 Tanimoto, between its own top 100 (0.35) and random molecules (0.09).
  - CLAMP scores ECFP's top 100 at 0.73 cosine, between its own (0.85) and random molecules (0.52).
- **Queries differ a lot.** At k = 10, a quarter of the queries share at most 2 molecules and another quarter at least 6.

**The indexes change almost nothing.** The exact table, from `lazysim compare`, matches this one within 0.004 for shares and similarities and within 3% for positions. ECFP's index is exact. Its small differences come from ties: many molecules can share the k-th Tanimoto score, and FPSim2 and the exact search keep different ones among them.

### All pairs

The share of the top k that both featurizers return, for the typical query, from what `lazysim search` returns. The last column gives the typical position of each featurizer's top 10 in the other's ranking: A's in B's, then B's in A's.

| A and B | k = 10 | k = 100 | k = 1,000 | Top 10, in the other's ranking |
|---|---|---|---|---|
| ECFP and SAND | 0.40 | 0.23 | 0.12 | 17 and 16 |
| CLAMP and ECFP | 0.40 | 0.18 | 0.08 | 21 and 23 |
| CLAMP and SAND | 0.30 | 0.15 | 0.06 | 36 and 35 |
| ECFP and Mol-JEPA | 0.20 | 0.12 | 0.10 | 178 and 182 |
| SAND and Mol-JEPA | 0.20 | 0.10 | 0.07 | 181 and 579 |
| CLAMP and Mol-JEPA | 0.20 | 0.10 | 0.06 | 198 and 913 |

- **CLAMP, ECFP and SAND agree on the closest analogues.** Each pair shares 3 or 4 of the top 10, and the typical molecule of each top 10 sits within the top 40 of the other's ranking.
- **ECFP and SAND agree the most,** at every k.
- **Mol-JEPA agrees least with the others.** It shares 2 of the top 10 with each of them, and the typical molecule of its top 10 sits between 182nd and 913th in their rankings.
- **Every pair agrees far more than chance.** A random molecule sits at 677,554th, and two random top 10s of 1.36M molecules share almost nothing.
- **The exact rankings give the same picture** for every pair: shares within 0.01, similarities within 0.004 and typical positions within 7%.

### Adding a pair

```bash
python scripts/compare_featurizers.py <library> -f <a> -f <b> --save
```

- The library needs both featurizers' indexes and embeddings (`lazysim fetch NAME -f A -f B --embeddings`).
- `--save` adds the pair to `src/lazy_sim/results/<library>.json`, one file per library holding every pair compared. Commit that file and `lazysim compare` shows the new pair.
- On the Ersilia Reference Library, each pair took 2–3 minutes on 6 CPUs and 4–8 GB of memory.
