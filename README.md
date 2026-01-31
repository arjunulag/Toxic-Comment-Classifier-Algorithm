# 🛡️ Toxic Comment Classifier

A deep learning model for detecting toxic comments using **Transfer Learning** with **DistilBERT**. This project fine-tunes a pre-trained transformer model to classify text as toxic or non-toxic with high accuracy.

---

## 📋 Overview

This classifier leverages the power of transfer learning by fine-tuning [DistilBERT](https://huggingface.co/distilbert-base-uncased) (a lighter, faster version of BERT with 66M parameters) on a toxic comments dataset. The model achieves strong performance in identifying harmful, offensive, or toxic language in text.

### Key Features

- **Transfer Learning**: Uses pre-trained DistilBERT for superior language understanding
- **Binary Classification**: Classifies comments as Toxic or Non-Toxic
- **Multiple Interfaces**: CLI tool, interactive mode, and Streamlit web app
- **GPU Support**: Automatically uses CUDA if available for faster inference
- **Production Ready**: Saved model can be easily deployed

---

## 🚀 Quick Start

### Installation

1. Clone the repository and navigate to the project directory

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Download the Dataset

This project uses the **Jigsaw Unintended Bias in Toxicity Classification** dataset from Kaggle.

1. Go to the [Kaggle competition page](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/data)
2. Download `train.csv`
3. Rename it to `data.csv` and place it in the project root

**Alternative**: Using Kaggle CLI:
```bash
# Install Kaggle CLI (if not already installed)
pip install kaggle

# Download the dataset (requires Kaggle API credentials)
kaggle competitions download -c jigsaw-unintended-bias-in-toxicity-classification -f train.csv
mv train.csv data.csv
```

> **Note**: You'll need a Kaggle account and API token. See [Kaggle API docs](https://www.kaggle.com/docs/api) for setup.

### Training the Model

Train the model on your dataset:

```bash
python train.py
```

The training script will:
- Load and preprocess `data.csv`
- Fine-tune DistilBERT for 3 epochs
- Save the best model to `toxic_classifier_model/`
- Display training metrics (accuracy, F1, AUC-ROC)

### Making Predictions

**Demo mode** (sample predictions):
```bash
python predict.py --demo
```

**Single prediction**:
```bash
python predict.py --text "Your comment here"
```

**Interactive mode**:
```bash
python predict.py --interactive
```

### Web Application

Launch the Streamlit web interface:
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
Classify/
├── train.py                    # Training script
├── predict.py                  # CLI prediction tool
├── app.py                      # Streamlit web application
├── data.csv                    # Training dataset (download separately)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── toxic_classifier_model/     # Saved model directory (generated after training)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── special_tokens_map.json
└── README.md
```

> **Note**: `data.csv` and `toxic_classifier_model/` are not included in the repository due to size. See download instructions above.

---

## ⚙️ Configuration

Training parameters can be modified in the `Config` class within `train.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_NAME` | `distilbert-base-uncased` | Pre-trained model to fine-tune |
| `MAX_LENGTH` | `256` | Maximum token sequence length |
| `BATCH_SIZE` | `16` | Training batch size |
| `EPOCHS` | `3` | Number of training epochs |
| `LEARNING_RATE` | `2e-5` | AdamW learning rate |
| `WARMUP_RATIO` | `0.1` | Warmup steps ratio |
| `TOXICITY_THRESHOLD` | `0.5` | Threshold for toxic label |
| `TEST_SIZE` | `0.2` | Validation split ratio |

---

## 📊 Dataset Format

The training data (`data.csv`) should contain:

| Column | Description |
|--------|-------------|
| `comment_text` | The text content to classify |
| `target` | Toxicity score (0.0 - 1.0). Values ≥ 0.5 are labeled as toxic |

---

## 🔧 Model Architecture

```
DistilBERT (Pre-trained)
    │
    ├── 6 Transformer Layers
    ├── 768 Hidden Dimensions
    ├── 12 Attention Heads
    │
    └── Classification Head (2 classes)
         ├── Dropout
         └── Linear Layer → [Non-Toxic, Toxic]
```

### Training Process

1. **Tokenization**: Text is tokenized using DistilBERT's WordPiece tokenizer
2. **Fine-tuning**: All model layers are trained with gradient clipping
3. **Optimization**: AdamW optimizer with linear warmup scheduler
4. **Evaluation**: Model selection based on best validation F1 score

---

## 📈 Metrics

The model is evaluated using:

- **Accuracy**: Overall classification accuracy
- **Precision**: Fraction of true positives among predicted positives
- **Recall**: Fraction of true positives among actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve

---

## 💻 API Usage

### Python Integration

```python
from predict import ToxicCommentClassifier

# Initialize classifier
classifier = ToxicCommentClassifier('toxic_classifier_model')

# Single prediction
result = classifier.predict("This is a test comment")
print(result)
# {
#     'text': 'This is a test comment',
#     'prediction': 'Non-Toxic',
#     'is_toxic': False,
#     'confidence': 0.98,
#     'toxic_probability': 0.02,
#     'non_toxic_probability': 0.98
# }

# Simple classification
label = classifier.classify("Another comment")
print(label)  # 'Non-Toxic' or 'Toxic'

# Batch prediction
results = classifier.predict_batch(["Comment 1", "Comment 2", "Comment 3"])
```

---

## 🖥️ Web Interface

The Streamlit app (`app.py`) provides:

- Modern, dark-themed UI
- Real-time toxicity analysis
- Confidence scores and probability distribution
- Example buttons for quick testing

---

## 📦 Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- CUDA (optional, for GPU acceleration)

See `requirements.txt` for full dependency list.

---

## 🔬 Technical Details

### Tokenization
- **Tokenizer**: DistilBERT WordPiece
- **Max Length**: 256 tokens
- **Padding**: Max length padding
- **Truncation**: Enabled for long texts

### Training
- **Optimizer**: AdamW (ε = 1e-8)
- **Scheduler**: Linear warmup + decay
- **Gradient Clipping**: Max norm = 1.0
- **Loss Function**: Cross-entropy

### Model Selection
- Best model is selected based on validation F1 score
- Model weights saved in SafeTensors format

---

## 📝 License

This project is for educational and research purposes.

---

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for the pre-trained models
- [Jigsaw/Google](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification) for the toxic comments dataset
- [Streamlit](https://streamlit.io/) for the web framework
