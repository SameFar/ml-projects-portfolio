<div align="center">

# 🌍 Character-Level RNN for Name Origin

**Reads a surname letter by letter and guesses its language of origin.**

![Deep Learning](https://img.shields.io/badge/🟣_DEEP_LEARNING-8957e5?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)

</div>

---

Give it a name like `Nakamura` or `O'Brien` and it predicts which of 18 languages it most likely came from — one character at a time. No pretrained embeddings, no tokenizer, just a plain RNN reading the name letter by letter.

## 🧠 How it works

1. **Data** — `data/names/` has one `.txt` file per language (18: Arabic, Chinese, Czech, Dutch, English, French, German, Greek, Irish, Italian, Japanese, Korean, Polish, Portuguese, Russian, Scottish, Spanish, Vietnamese), one surname per line.
2. **Preprocessing** — each name is normalized to a fixed 58-character alphabet (52 upper/lowercase ASCII letters plus `` .,;'_``). Accents are stripped to the nearest ASCII (`speak_merican()`), and anything still outside is mapped to `_`. Each name becomes a sequence of one-hot vectors.
3. **Model** (`src/RNN.py`) — a single `nn.RNN` (input 58, hidden 128); its final hidden state goes through a Linear layer to 18 outputs, then `LogSoftmax`.
4. **Training** (`src/train.py`) — `NLLLoss` with plain SGD (lr `0.05`), 100 epochs, batch 64, an 85/15 split, gradient-norm clipping at 3. Validation accuracy prints every 5 epochs; weights save to `model/model.pt`, and a loss curve is plotted at the end (shown, not saved).
5. **Inference** (`src/guess.py`) — loads `model/model.pt` into a fresh `CharRNN` and returns the top-scoring language.

A trained checkpoint already ships in `model/model.pt`, so you can jump straight to inference.

## ⚠️ A quirk worth knowing

The language-to-index mapping used in training comes from `list(set(...))` over the filenames, and Python randomizes string hashing per process — so that set's iteration order isn't stable across runs. `src/guess.py` hardcodes its own fixed order of the 18 languages. If you **retrain**, the learned class indices may not line up with `guess.py`'s list, so predictions could come out mislabeled. The shipped `model/model.pt` is fine as long as nobody retrains it, but it's a real fragility in the code as written.

## 🚀 Getting started

```bash
uv sync
uv run main.py
```

You'll be prompted:
- `y` — trains a fresh model, plots the loss curve, saves to `model/model.pt`, then exits (doesn't loop back).
- `n` — asks for a name, prints the predicted origin from the saved model, then asks again.
- anything else — exits.
