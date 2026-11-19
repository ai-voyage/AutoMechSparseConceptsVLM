# Automating Mechanistic Sparse Concept Discovery in Vision-Language Models

**Nima Mirnateghi, Syed Mohammed Shamsul Islam, Syed Afaq Ali Shah**
*DICTA 2026*

[Paper](#) · [Artefacts](#pretrained-artefacts) · [Citation](#citation)

This repository contains the official implementation of our DICTA paper. We introduce a framework
that automatically builds data-specific concept vocabularies with a vision-capable language model,
verifies them with an open-vocabulary detector, and uses them to name the neurons of a sparse
autoencoder (SAE) trained on CLIP image features. The named neurons form an interpretable concept
bottleneck for classification.

Our implementation extends [Discover-then-Name](https://github.com/neuroexplicit-saar/discover-then-name)
(Rao et al., ECCV 2024), included in `src/dncbm` and `src/sparse_autoencoder`. Our contributions are
the multimodal concept generation and filtering pipeline (`src/concept_gen`, `scripts/concepts`) and a
modified SAE sparsity loss (`src/sparse_autoencoder/loss/learned_activations_l1.py`). The original
licences are retained in `NOTICE`.

<p align="center">
  <img src="assets/framework.png" width="85%" alt="Overview of the framework">
</p>

## Method overview

| Stage | Description |
|---|---|
| Concept generation | GPT-4o mini is prompted with five training images per class and few-shot text prompts. |
| Concept filtering | Grounding-DINO keeps only the concepts it detects in those images (box threshold 0.15, text threshold 0.25). |
| Sparse concept discovery | An SAE with thresholded activations and a square-root sparsity penalty is trained on CLIP RN50 features of CC3M. |
| Neuron naming | Each SAE neuron is assigned the concept whose CLIP text embedding is closest to its dictionary vector. |
| Classification | A sparse linear probe on the SAE activations yields a concept bottleneck model. |

## Installation

The code requires Python 3.10 and PyTorch 2.1.2.

```bash
git clone https://github.com/<user>/AutoMechSparseConceptsVLM.git
cd AutoMechSparseConceptsVLM
conda env create -f environment.yml
conda activate automech-sparse-concepts-vlm
```

Alternatively, in an existing environment:

```bash
pip install -r requirements.txt
pip install -e .
```

If the default PyTorch wheel does not match your CUDA version, install PyTorch 2.1.2 for your
system first.

### External repositories

The following projects are used alongside this repository and must be obtained separately:

| Project | Used for | Required |
|---|---|---|
| [Grounding-DINO](https://github.com/IDEA-Research/GroundingDINO) | Filtering generated concepts | Concept generation only |
| [img2dataset](https://github.com/rom1504/img2dataset) | Downloading CC3M images | Data preparation only |

**Grounding-DINO.** Install the official release inside the repository folder (the location is
set in `configs/paths.yaml`):

```bash
git clone https://github.com/IDEA-Research/GroundingDINO.git
pip install -e GroundingDINO
mkdir -p GroundingDINO/weights
wget -P GroundingDINO/weights https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

**img2dataset.**

```bash
pip install img2dataset
```

### Other requirements

**OpenAI API access** is needed only for concept generation. Create an API key in your own
OpenAI account and set it as an environment variable:

```bash
export OPENAI_API_KEY=<your-key>
```

The key is read only from this variable. Do not store it in code or configuration files.

**LaTeX** is required to render the visualisations, as the plotting scripts use `text.usetex`.

### Configuration

All file locations are defined in `configs/paths.yaml`. Relative paths are resolved against the
repository root. To use a different file, set `DNCBM_PATHS=/path/to/paths.yaml`. Run all commands
from the repository root.

## Pretrained artefacts

Trained models, activations and embeddings are hosted separately. Sizes and target paths are
listed in [`docs/artefacts.md`](docs/artefacts.md).

| Artefact | Download |
|---|---|
| SAE checkpoint, neuron names and linear probes | Link to be added |
| Concept vocabulary embeddings | Link to be added |
| CLIP RN50 activations (CC3M, Places365, CIFAR10, CIFAR100) | Link to be added |

## Data preparation

### CC3M

Download `Train_GCC-training.tsv` and `Validation_GCC-1.1.0-Validation.tsv` from
[Conceptual Captions](https://ai.google.com/research/ConceptualCaptions/download), rename them to
`cc3m_training.tsv` and `cc3m_validation.tsv`, and download the images as webdataset shards:

```bash
for split in training validation; do
    sed -i '1s/^/caption\turl\n/' cc3m_${split}.tsv
    img2dataset --url_list cc3m_${split}.tsv --input_format "tsv" \
        --url_col "url" --caption_col "caption" --output_format webdataset \
        --output_folder data/CC3M_TAR/${split} \
        --processes_count 16 --thread_count 64 --image_size 256
done
```

Some URLs are no longer available, so the number of downloaded images may be lower than in our
experiments.

### Evaluation datasets

Place Places365 (standard, 256×256) under `data/datasets/` in the `torchvision` layout. CIFAR10
and CIFAR100 are downloaded automatically.

### CLIP activations

```bash
python scripts/data/save_cc3m_features.py --img_enc_name clip_RN50
for ds in places365 cifar10 cifar100; do
    python scripts/data/save_probe_features.py --img_enc_name clip_RN50 --probe_dataset $ds
done
```

Each training set is split into `train` (90%) and `train_val` (10%).

## Training pipeline

The commands below use the settings reported in the paper:

```bash
SAE="--img_enc_name clip_RN50 --lr 5e-4 --l1_coeff 3e-5 --expansion_factor 4 \
     --num_epochs 200 --resample_freq 10 --train_sae_bs 4096"
PROBE="--probe_lr 0.01 --probe_train_bs 512 --probe_epochs 200 \
       --probe_sparsity_loss L1 --probe_sparsity_loss_lambda 0.1"
```

### 1. Concept sets

The concept sets used in the paper are provided in `data/concept_sets/<dataset>/`. To generate
new ones:

```bash
python scripts/concepts/generate_concepts.py --dataset places365
python scripts/concepts/build_vocab.py --dataset places365 --concept_dir outputs/concept_sets/places365
```

Use `generate_concepts.py --dry-run` to check data loading and output format without calling the
OpenAI API or Grounding-DINO.

Then embed each vocabulary with the CLIP text encoder:

```bash
for ds in places365 cifar10 cifar100; do
    python scripts/concepts/embed_vocab.py --img_enc_name clip_RN50 --vocab_name $ds
done
```

### 2. Sparse autoencoder

```bash
python scripts/sae/train_sae.py $SAE --l1_threshold 0.05 --ckpt_freq 0 --val_freq 1
```

### 3. Neuron naming

```bash
python scripts/naming/assign_names.py $SAE --probe_dataset places365
```

### 4. Concept bottleneck classifier

```bash
for split in train train_val val; do
    python scripts/probe/save_concept_strengths.py $SAE --probe_dataset places365 --probe_split $split
done
python scripts/probe/train_linear_probe.py $SAE $PROBE --probe_dataset places365
```

Replace `places365` with `cifar10` or `cifar100` for the other datasets. Results are appended to
`outputs/probe/probe_results.csv`.

## Visualisations

Each visualisation runs in two steps: a `dump` script computes and saves the data, and a `plot`
script renders it. Both take the same arguments. Run them after the training pipeline, or after
downloading the pretrained artefacts. Outputs are written to `outputs/analysis/`.

| Visualisation | Shows | Scripts |
|---|---|---|
| Task agnosticity | Top-activating images and assigned names for selected neurons, across datasets | `vis_task_agnosticity_{dump,plot}.py` |
| Local explanations | The concepts that contribute most to the prediction for individual images | `vis_local_explanations_{dump,plot}.py` |
| Meta-clustering | Groups of semantically related neurons | `vis_meta_clustering_{dump,plot}.py` |

**Task agnosticity**

```bash
python scripts/figures/vis_task_agnosticity_dump.py $SAE --probe_dataset <dataset> --probe_split val
python scripts/figures/vis_task_agnosticity_plot.py $SAE --probe_dataset <dataset> --probe_split val
```

**Local explanations** (pass a text file of image indices to explain; an example is provided in
`data/local_explanations/`)

```bash
python scripts/figures/vis_local_explanations_dump.py $SAE $PROBE --probe_dataset <dataset> \
    --probe_split val --img_idxs_file <indices.txt>
python scripts/figures/vis_local_explanations_plot.py $SAE $PROBE --probe_dataset <dataset> --probe_split val
```

**Meta-clustering**

```bash
python scripts/figures/vis_meta_clustering_dump.py $SAE --probe_dataset <dataset> --probe_split val
python scripts/figures/vis_meta_clustering_plot.py $SAE --probe_dataset <dataset> --probe_split val
```

Replace `<dataset>` with `places365`, `cifar10` or `cifar100`.

## Repository structure

```
AutoMechSparseConceptsVLM/
├── configs/     File paths and the hyperparameter search space
├── data/        Concept sets, class names and example image indices
├── docs/        Details of the separately hosted artefacts
├── results/     Result files from our experiments
├── scripts/     Entry points: data, concepts, sae, naming, probe, figures
└── src/         Packages: dncbm, sparse_autoencoder, clip, concept_gen
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{mirnateghi2026automating,
  title     = {Automating Mechanistic Sparse Concept Discovery in Vision-Language Models},
  author    = {Mirnateghi, Nima and Islam, Syed Mohammed Shamsul and Shah, Syed Afaq Ali},
  booktitle = {International Conference on Digital Image Computing: Techniques and Applications (DICTA)},
  year      = {2026},
  pages     = {PAGES},
  doi       = {DOI}
}
```

## Licence

This project is released under the MIT Licence. See `LICENSE`. Third-party components retain
their original licences, listed in `NOTICE`.
