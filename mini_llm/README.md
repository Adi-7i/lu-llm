# Mini LLM

A lightweight, educational implementation of a **mini language model** built from scratch using Python and PyTorch. This project demonstrates core LLM concepts—tokenization, embeddings, transformer layers, loss computation, back‑propagation, and inference—on a tiny dataset, making it ideal for learning and experimentation.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Training](#training)
- [Inference (Chat Interface)](#inference-chat-interface)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The **Mini LLM** is a minimalistic language model that:
- Implements a custom tokenizer.
- Uses an embedding layer followed by a stack of transformer blocks.
- Trains on a small, user‑provided text corpus.
- Provides a simple command‑line chat interface for inference.

The codebase is intentionally kept small and well‑documented to serve as a teaching aid for anyone interested in the inner workings of large language models.

---

## Features

- **Pure PyTorch implementation** – no external ML libraries beyond PyTorch.
- **Modular design** – clear separation of tokenization, model architecture, training loop, and inference utilities.
- **Configurable hyper‑parameters** – easily adjust model size, number of layers, learning rate, etc.
- **Minimal dependencies** – only Python 3.9+ and PyTorch are required.
- **Extensible** – a solid foundation for adding features like positional encodings, attention visualizations, or larger datasets.

---

## Project Structure

```
mini_llm/
├─ data/                # Sample dataset (plain‑text files)
├─ model/
│   ├─ __init__.py
│   ├─ layers.py        # Transformer block implementation
│   └─ model.py         # Full model definition (embedding + layers)
├─ training/
│   ├─ __init__.py
│   ├─ loss.py          # Custom loss function (e.g., CrossEntropy)
│   └─ train.py         # Training loop
├─ utils/
│   └─ __init__.py
├─ .gitignore           # Git ignore rules (generated)
├─ README.md            # Project documentation (this file)
└─ requirements.txt     # Python dependencies
```

---

## Setup & Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mini_llm
   ```
2. **Create a virtual environment** (optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Verify the installation**
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

---

## Training

The training script is located at `training/train.py`. A typical command looks like:

```bash
python training/train.py \
    --data_path data/dataset.txt \
    --epochs 10 \
    --batch_size 32 \
    --lr 3e-4 \
    --model_dir checkpoints/
```

Adjust the arguments as needed. Training progress and loss are logged to the console and saved checkpoints.

---

## Inference (Chat Interface)

After training, you can start a simple chat interface:

```bash
python -m mini_llm.inference \
    --model_path checkpoints/latest.pt
```

Type a prompt and press **Enter** to receive a generated response. Use `Ctrl+C` to exit.

---

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests. When contributing:
- Follow the existing coding style.
- Add docstrings and comments for new functions.
- Update the `README.md` if you add major features.

---

## License

This project is licensed under the **MIT License** – see the `LICENSE` file for details.

---

*Created with ❤️ by the Mini LLM team.*
