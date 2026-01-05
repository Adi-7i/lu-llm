# Mini LLM (lu-llm)

A **lightweight, educational** implementation of a mini language model built from scratch using **Python** and **PyTorch**. This project showcases core LLM concepts—tokenization, embeddings, transformer layers, loss computation, back‑propagation, and inference—on a tiny dataset, making it ideal for learning and experimentation.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Training](#training)
- [Inference (Chat)](#inference-chat)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The **Mini LLM** provides a clear, modular codebase that:
- Implements a custom tokenizer.
- Uses an embedding layer followed by a stack of transformer blocks.
- Trains on a small, user‑provided text corpus.
- Offers a simple command‑line chat interface for inference.

All components are written in pure PyTorch with minimal external dependencies, making the inner workings easy to follow.

---

## Features
- **Pure PyTorch** – no heavyweight ML libraries.
- **Modular design** – separate modules for tokenization, model architecture, training, and inference.
- **Configurable hyper‑parameters** – adjust model size, layers, learning rate, etc.
- **Minimal dependencies** – only Python 3.9+ and PyTorch.
- **Extensible** – a solid base for adding positional encodings, attention visualizations, larger datasets, etc.

---

## Project Structure
```
mini_llm/
├─ data/                # Sample dataset (plain‑text files)
├─ model/
│   ├─ __init__.py
│   ├─ layers.py        # Transformer block implementation
│   └─ model.py         # Full model (embedding + layers)
├─ training/
│   ├─ __init__.py
│   ├─ loss.py          # Loss function (Cross‑Entropy)
│   └─ train.py         # Training loop
├─ inference/
│   ├─ __init__.py
│   └─ chat.py          # Simple CLI chat interface
├─ utils/
│   └─ __init__.py
├─ .gitignore           # Generated ignore file
├─ README.md            # This documentation
└─ requirements.txt     # Python dependencies
```

---

## Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/Adi-7i/lu-llm.git
   cd lu-llm
   ```
2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Verify PyTorch installation**
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

---

## Training
Run the training script with your dataset:
```bash
python training/train.py \
    --data_path data/dataset.txt \
    --epochs 10 \
    --batch_size 32 \
    --lr 3e-4 \
    --model_dir checkpoints/
```
Adjust hyper‑parameters as needed. Checkpoints are saved in `checkpoints/`.

---

## Inference (Chat)
After training, start the interactive chat:
```bash
python -m inference.chat \
    --model_path checkpoints/latest.pt
```
Type a prompt and press **Enter** to receive a response. Use `Ctrl+C` to exit.

---

## Contributing
Contributions are welcome! Please:
- Follow the existing coding style.
- Add docstrings for new functions.
- Update this `README.md` when adding major features.
- Open an issue or pull request on GitHub.

---

## License
This project is licensed under the **MIT License** – see the `LICENSE` file for details.

---

*Created with ❤️ by the Mini LLM team.*
