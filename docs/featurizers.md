# Featurizers

How each lazy-sim featurizer represents molecules, where it comes from, how it was checked against the original code, and how well it searches.

| `-f` | Captures | Similarity | Engine | Runs on |
|---|---|---|---|---|
| `clamp` | Bioactivity-aware embedding (768 numbers) | Cosine | [turbovec](https://pypi.org/project/turbovec/), 4-bit (approximate) | RDKit + numpy |
| `ecfp` | Substructures, ECFP6 (2,048 bits) | Tanimoto | [FPSim2](https://github.com/chembl/FPSim2) (exact) | RDKit |
| `sand` | 3D shape (512 numbers) | Cosine | turbovec, 4-bit (approximate) | RDKit + numpy |
| `moljepa` | Biological, phenotypic and quantum context predicted from the graph (512 numbers); **non-commercial use only** | Cosine | turbovec, 4-bit (approximate) | RDKit + numpy |
| `monroe` | Structure and stereochemistry learned from quantum-chemical and bioassay data (720 numbers) | Cosine | turbovec, 4-bit (approximate) | RDKit + numpy |

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

## Mol-JEPA

### Source

- **Authors:** Boehringer Ingelheim.
  - Preprint on [arXiv](https://arxiv.org/abs/2608.22642) (2026).
  - Code at [Boehringer-Ingelheim/mol-jepa](https://github.com/Boehringer-Ingelheim/mol-jepa).
  - Weights on Hugging Face at [Flogrammer/Mol-JEPA](https://huggingface.co/Flogrammer/Mol-JEPA).
- **Licence: CC BY-NC 4.0, non-commercial use only.**
  - The licence covers the code and the weights.
  - Commercial use needs the authors' written permission.
  - lazy-sim prints this notice whenever it needs the weights.
- **On the Ersilia Model Hub:** [eos9q2i](https://github.com/ersilia-os/eos9q2i).
- **What it captures:**
  - Mol-JEPA was trained by masking whole modalities across 4.69M compounds: a molecular graph, ECFP4, quantum-chemistry and structural embeddings, descriptors, and bioactivity labels.
  - Per the authors, its embedding carries predicted biological, phenotypic and quantum context.
  - From a SMILES, only the graph is observed; the other modalities are treated as missing.

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
   - Each has 8 heads, each 512 wide, averaged; bond features are added to the keys and values.
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
  - Downloaded once to `~/lazysim/models/`; `LAZYSIM_MOLJEPA_WEIGHTS` points to a local copy instead.
  - The sha256 is checked at every load and recorded in each library.

### Checked against the original

- **The reference:** Reference Library molecules embedded with Mol-JEPA's own code, run offline from the pinned snapshot that eos9q2i stores.
  - The setup was torch 2.8, transformers 4.50.3, molfeat 0.10.1 and RDKit 2026.3.4.
- **Result on 10,000 random molecules:**
  - 9,999 graphs are identical to molfeat's, features and edge order, and their embeddings agree within 1e-6 (median 8e-8).
  - The remaining molecule differs because of RDKit's version: in a furan inside a large ring system, RDKit 2026.3.4 sees 5 aromatic atoms and 2026.3.6, which lazy-sim uses, sees none. Its cosine to the original is 0.90. SAND's check flagged the same molecule.
- **Kept as a test:** `tests/test_cli.py::test_moljepa_matches_original`.
  - It embeds 20 molecules and compares them with the original's embeddings (`tests/data/moljepa_reference.npy`).
  - Those reference embeddings come from [`scripts/export_moljepa_reference.py`](../scripts/export_moljepa_reference.py), which runs Mol-JEPA's own code offline from the pinned snapshot. It's the counterpart of the other featurizers' export scripts; there are no weights to export.
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
- **Indexing:** the Ersilia Reference Library would take about 5–6 hours.

## Monroe

### Source

- **Authors:** Banaszewski & Fitzgibbon.
  - Preprint on [arXiv](https://doi.org/10.48550/arXiv.2608.18982) (2026).
  - Code and weights at [blazejba/Monroe](https://github.com/blazejba/Monroe) (commit `57238ed`).
- **Licence:**
  - Monroe is MIT, Copyright (c) 2026 Blazej Banaszewski.
  - Its attention layers are adapted from [GRIT](https://github.com/LiamMa/GRIT) (MIT, Copyright (c) 2023 Liheng Ma).
  - Both notices are kept next to lazy-sim's copy of the weights.
- **On the Ersilia Model Hub:** [eos8gop](https://github.com/ersilia-os/eos8gop). lazy-sim reproduces what eos8gop serves, including its fixed conformer seed.
- **What it captures:**
  - A graph transformer pretrained on 81 million PM6 quantum-chemistry calculations and 1,089 PubChem bioassays.
  - Extra edges encode E/Z and R/S configuration, so stereoisomers get different inputs.
  - The authors pair the embedding with a TabPFN head for few-shot property prediction; only the encoder is used here.

### What lazy-sim computes

**Molecule.**
- The SMILES is converted to an InChI and the molecule is rebuilt from it, as Monroe does. Then hydrogens are added and RDKit's legacy stereo perception is applied.
- When the InChI round trip fails, lazy-sim builds the molecule from the SMILES instead, so every library row gets an embedding. eos8gop returns NaN for these molecules; this is the one deliberate difference.

**Conformer.**
- One RDKit ETKDGv3 conformer: seed 42, random starting coordinates, chirality enforced, up to 500 iterations.
- It is optimised with MMFF94s for up to 200 iterations, or with UFF where MMFF94s has no parameters.
- If both fail, lazy-sim uses 2D coordinates.
- The 3D coordinates only enter as distances, between bonded or stereo-linked atoms and from each atom to the origin.

**Graph.**
- **Per atom:** 8 numbers (radical electrons, electronegativity, first ionisation energy, covalent radius, melting point, formal charge, van der Waals radius, mass) and 11 categories (implicit and total valence, hybridisation, element, H count, aromatic, in ring, CIP label, period, degree, group).
- **Per bond:** 4 categories (conjugated, in ring, type, stereo).
- **Extra edges:** E/Z edges around stereo double bonds, and R/S edges around four-coordinate stereocentres.
- **Virtual node:** each molecule gets one, linked to every atom.

**Network.** 10 GRIT layers, 720 wide with 10 heads, 58.7M parameters, run in numpy.
- **Inputs:** RBF expansions of the atom numbers and edge distances, plus random-walk encodings over 16 steps of the bond graph.
- **Each layer:** attention over each node's incoming edges, with edge-dependent weights and biases; degree scaling; batch normalisation (fixed statistics); and a feed-forward block.
- **Readout:** attention pooling with 10 heads over the atoms and the virtual node gives the 720 numbers.

**Differences from the original.**
- lazy-sim L2-normalises the output so cosine is a dot product.
- It embeds molecules in batches; the original embeds one at a time. The results are the same, because the batch normalisation uses fixed statistics.

The code is in [`src/lazy_sim/featurizers/monroe.py`](../src/lazy_sim/featurizers/monroe.py), with the periodic-table values in `_monroe_elements.py`.

### Weights

**Original:** `checkpoint/weights.pt` from blazejba/Monroe (300,057,823 bytes, sha256 `df996e8e…`), which eos8gop stores with eosvc. Only the encoder is kept.

**Conversion:** [`scripts/export_monroe_weights.py`](../scripts/export_monroe_weights.py), which needs torch and is for maintainers. With eos8gop's code, it also writes the reference embeddings for the tests.

**The converted file:** `monroe_encoder_v1.npz`.
- It holds 338 arrays under Monroe's own parameter names.
- Size: 234,838,658 bytes.
- sha256: `39fffb2a21f44d72be6c1ca3b37fd24ad0300e3a13c04d5b564d62a97d1a02ba`.

**Hosting:** on Ersilia's public S3 at `lazy-sim/data/monroe/`, next to Monroe's `LICENSE` and `NOTICE`.

**Download and checks:** the same as CLAMP. It is downloaded once to `~/lazysim/models/` (`LAZYSIM_MONROE_WEIGHTS` points to a local copy), and its sha256 is checked at every load and recorded in each library.

### Checked against the original

- **The reference:** 10,000 random Reference Library molecules, embedded with eos8gop's own code: torch 2.14, torch_geometric 2.8 and RDKit 2026.3.6, the version lazy-sim also uses.
- **Result:**
  - For all 9,999 molecules the original can embed, the graphs are identical, 3D conformer coordinates included.
  - The embeddings agree within 2.1e-6 (median 7.5e-8).
  - The remaining molecule fails eos8gop's InChI round trip, and lazy-sim embeds it from its SMILES.
- **Kept as a test:** `tests/test_cli.py::test_monroe_matches_original`.
  - It embeds 20 molecules and compares them with eos8gop's embeddings (`tests/data/monroe_reference.npy`).
  - One of them, cisplatin, uses the 2D-coordinate fallback.
  - The largest allowed difference is 1e-5.
- **Reproducibility:** conformers depend on RDKit's version, so the embeddings match only with RDKit 2026.3.6. Another CPU architecture, such as ARM, may also give slightly different conformers.

### Speed

Monroe is the heaviest featurizer: 10 layers over about 4 edges per atom, each 720 wide, plus a 3D conformer per molecule. On a 16-thread laptop:

| | Molecules per second |
|---|---|
| Network, lazy-sim (numpy, one process) | ~18–20 |
| Conformers and graphs, lazy-sim (16 worker processes) | ~84 (about 0.2 s per molecule each) |
| Original (eos8gop, one molecule at a time, 1 thread per process) | 3.4 per process |

- **Where the network's time goes:** matrix products take about 58%, on all threads; the rest is per-edge numpy work on one thread.
- **Queries:** conformers are slow, so Monroe builds query conformers in worker processes from 100 queries (`POOL_MIN_QUERIES`; the other featurizers wait for 5,000). 300 queries take about 37 s.
- **Indexing:** the Ersilia Reference Library would take about a day.
