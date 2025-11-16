# AdaAlign: Adaptive Alignment for Sketch-Based Image Retrieval

**Official PyTorch Implementation**

## 📌 Overview

AdaAlign is an adaptive cross-domain alignment framework for
**Sketch-Based Image Retrieval (SBIR)**.\
The method improves sketch--image matching by integrating:

-   Lightweight **PEFT modules** (Adapter / LoRA / Prompt learning)\
-   Cross-modal **visual--textual distillation**\
-   Domain-adaptive feature refinement for abstract sketches\
-   Multiple teacher encoders (ResNet, DINO, ViT, CLIP)\
-   Fully reproducible training/testing pipelines

This repository provides model implementations, training scripts, and
visualization tools.

------------------------------------------------------------------------

## 📂 Project Structure

    modeling/
      adapter.py
      resnet.py
      senet.py
      vision_transformer.py
      text_encoder.py
      utils_dino.py
      model.py
      clip/
          clip.py
          model.py
          simple_tokenizer.py
          bpe_simple_vocab_16e6.txt.gz

    src/
      heat_map.py
      t-SNE.py
      *.ipynb
      scripts/
          train/
          test/
          train_new/

    utils/
      dataset.py
      loss.py
      model.py
      evaluate.py
      text_utils.py
      tools.py

    results/
    checkpoints/
    logs/
    train.py
    test.py
    options.py

------------------------------------------------------------------------

## 🔧 Installation

### Environment

``` bash
conda create -n adaalign python=3.8 -y
conda activate adaalign
```

### Dependencies

``` bash
pip install -r requirements.txt
```

For CLIP models:

``` bash
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

------------------------------------------------------------------------

## 📁 Datasets

Supported datasets:

  Dataset     Folder
  ----------- -------------------
  Sketchy     `data/sketchy/`
  TU-Berlin   `data/tuberlin/`
  QuickDraw   `data/quickdraw/`

Directory example:

    data/
    ├── sketchy/
    ├── tuberlin/
    └── quickdraw/

------------------------------------------------------------------------

## 🏋️ Training

### Default training

``` bash
python train.py --config options.py
```

### Example (Sketchy)

``` bash
python train.py     --dataset sketchy     --batch_size 64     --lr 5e-5     --epochs 40     --teacher RN50     --kd_weight 1.0
```

Training scripts (recommended):

    src/scripts/train/*.sh

------------------------------------------------------------------------

## 🔍 Evaluation

``` bash
python test.py     --checkpoint checkpoints/model_best.pth     --dataset sketchy     --topk 200
```

------------------------------------------------------------------------

## 🎨 Visualization

Tools for analyzing the model:

-   `src/t-SNE.py` -- t-SNE feature visualization\
-   `src/heat_map.py` -- heatmap and attention visualization\
-   Notebooks under `src/*.ipynb`

------------------------------------------------------------------------

## 🧠 Supported Backbones

**Teacher Models** - ResNet-50, SE-ResNet\
- DINO-S / DINO-B Vision Transformers\
- CLIP RN50 / ViT-B

**Student Models** - Adapter-based\
- LoRA\
- Prompt learning\
- Text-guided alignment

------------------------------------------------------------------------

## 📦 Checkpoints

Place all checkpoints under:

    checkpoints/

------------------------------------------------------------------------

## 📑 Citation

    @article{your2025adaalign,
      title={AdaAlign: Adaptive Alignment for Sketch-Based Image Retrieval},
      author={Your Name},
      journal={ArXiv},
      year={2025}
    }

------------------------------------------------------------------------

## 📬 Contact

For questions or issues, please open an Issue on GitHub.
