# Character-Level RNN for Name Origin Classification

This project guesses which language a surname comes from, one character at a time. Give it a name like `Nakamura` or `O'Brien` and it'll predict which of 18 languages it most likely originated from. No pretrained embeddings, no tokenizer — just a plain RNN reading the name letter by letter.

## How it works

1. **Data**: `data/names/` has one `.txt` file per language (18 files — Arabic, Chinese, Czech, Dutch, English, French, German, Greek, Irish, Italian, Japanese, Korean, Polish, Portuguese, Russian, Scottish, Spanish, Vietnamese), one surname per line.
2. **Preprocessing**: each name is normalized down to a fixed 58-character alphabet — the 52 upper/lowercase ASCII letters plus `` .,;'_``. Accented characters get stripped down to their closest ASCII equivalent (`speak_merican()` in `src/data_processing.py`), and anything still outside the alphabet gets mapped to `_`. Each name then becomes a sequence of one-hot vectors, one per character.
3. **Model** (`src/RNN.py`): a single `nn.RNN` layer (input size 58, hidden size 128) reads the character sequence, and its final hidden state goes through a Linear layer down to 18 outputs, followed by `LogSoftmax`.
4. **Training** (`src/train.py`): `NLLLoss` with plain SGD (lr `0.05`), 100 epochs by default, batch size 64, an 85/15 train/validation split, and gradient norm clipping at 3. Validation accuracy gets printed to the console every 5 epochs. Weights are saved to `model/model.pt`, and a loss curve is plotted (via matplotlib) at the end of training — it's shown on screen, not saved to disk.
5. **Inference** (`src/guess.py`): loads `model/model.pt` into a fresh `CharRNN` and returns the highest-scoring language for a given name.

A trained checkpoint already ships in `model/model.pt`, so you can jump straight to inference without training first.

## A quirk worth knowing about

The language-to-index mapping used during training comes from `list(set(...))` over the filenames in `data/names/` — and Python randomizes string hashing per process by default, so that set's iteration order isn't stable across runs. `src/guess.py`, meanwhile, hardcodes its own fixed list of the 18 languages in a specific order. If you retrain the model, there's no guarantee the class indices it learns will line up with `guess.py`'s hardcoded list, so predictions could come out mislabeled. This doesn't affect the shipped `model/model.pt` as long as nobody retrains it, but it's a real fragility in the code as written.

## Getting started

```bash
uv sync
uv run main.py
```

You'll be prompted:
- `y` — trains a fresh model on `data/names/`, plots the loss curve, and saves weights to `model/model.pt`. The program exits after training finishes (it doesn't loop back to the menu).
- `n` — asks for a name, prints the predicted language of origin using the saved `model/model.pt`, then asks again.
- anything else — exits.
