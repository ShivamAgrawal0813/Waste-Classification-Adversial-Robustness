# Waste Classification & Adversarial Robustness

A PyTorch-based waste classification system with adversarial robustness analysis. This project demonstrates clean model training, adversarial attacks (FGSM/PGD), and adversarial defense through robust training.

**Features:**
- ResNet-50/18 classifier for 6 waste categories (cardboard, glass, metal, paper, plastic, trash)
- FGSM and PGD adversarial attack implementations
- Adversarial training for robust models
- Interactive Streamlit web app for inference and visualization
- Complete evaluation pipeline with metrics and confusion matrices

## Project Structure

```
src/
├── config.py           # Hyperparameters and constants
├── dataset.py          # TrashNet dataset loader
├── train.py            # Clean model training
├── adv_train.py        # Adversarial training
├── attacks.py          # FGSM & PGD attacks
├── eval.py             # Evaluation script
├── inference.py        # Single-image inference
└── utils.py            # Helper functions

app/
└── streamlit_app.py    # Interactive web app

notebooks/
└── evaluate_models.ipynb  # Evaluation notebook with visualizations

models/                  # Trained checkpoints (.pth files)
outputs/                 # Metrics, plots, and predictions
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd "waste classifier 2.0"

# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
python verify_setup.py
```

### Dataset
Download [TrashNet dataset](https://github.com/garythung/trashnet) and place in:
```
data/trashnet/Garbage classification/Garbage classification/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

## Usage

### Training

```bash
# Train clean model
python src/train.py

# Train adversarially robust model
python src/adv_train.py
```

### Evaluation

```bash
# Evaluate both models and generate metrics
python src/eval.py
```

Outputs metrics and confusion matrices to `outputs/`.

### Web Demo

```bash
streamlit run app/streamlit_app.py
```

Upload an image to classify and visualize FGSM/PGD attacks side-by-side. The app displays:
- Clean predictions
- FGSM and PGD adversarial examples
- Perturbation visualization
- Model performance metrics

## Configuration

Hyperparameters are defined in `src/config.py`:
- **Model**: ResNet architecture, image size (224x224)
- **Training**: Learning rate (0.01), batch size (32), epochs (40-50)
- **Attacks**: FGSM ε=8/255, PGD ε=8/255, 20 steps
- **Dataset splits**: 70% train, 15% val, 15% test

## Adversarial Training

The project implements Madry-style adversarial training:
1. Generate PGD adversarial examples for each batch
2. Train on mixed clean + adversarial examples
3. Loss = 0.5 × L_clean + 0.5 × L_adversarial

This improves robustness against both FGSM and PGD attacks while maintaining reasonable clean accuracy.

## Results

After evaluation, the following files are generated:
- `outputs/metrics.json` - Clean and adversarial accuracies
- `outputs/confusion_clean.png` - Clean model confusion matrix
- `outputs/confusion_adv.png` - Adversarial model confusion matrix
- `outputs/accuracy_vs_eps.png` - Robustness curves

**Example metrics:**
- Clean model: ~93% clean accuracy, ~29% FGSM robustness, ~2% PGD robustness
- Adversarial model: ~87% clean accuracy, ~77% FGSM robustness, ~76% PGD robustness

## References

- TrashNet Dataset: [github.com/garythung/trashnet](https://github.com/garythung/trashnet)
- Madry et al. "Towards Deep Learning Models Resistant to Adversarial Attacks" (2018)
- Goodfellow et al. "Explaining and Harnessing Adversarial Examples" (2015)


