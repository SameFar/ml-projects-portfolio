<div align="center">

# 🎵 CNN-GRU Music Preference Model

**Guesses whether you'd like a song, straight from the audio.**

![Deep Learning](https://img.shields.io/badge/🟣_DEEP_LEARNING-8957e5?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![librosa](https://img.shields.io/badge/librosa-4D02A2?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

</div>

---

Give it a YouTube playlist of songs you like. It downloads them, listens with a pretrained audio model, and learns what your "taste" sounds like. Then hand it any YouTube link and it says **Liked** or **Disliked** based on how close the song sounds to that taste.

## 🧠 How it works

1. **Download** — `yt-dlp` grabs audio from a YouTube playlist, saving each track as a 30-second mp3 (mono, 16kHz).
2. **Listen** — each song is cut into 5-second chunks; a pretrained **PANN (Cnn14)** turns every chunk into an embedding describing what it sounds like. This backbone is frozen.
3. **Model time** — the chunk embeddings feed in order into a **GRU**, so the model picks up how a song changes over time, not just one snapshot.
4. **Learn your taste** — it only ever sees songs you like. It learns a single "center" point representing your average taste and pulls all liked songs toward it — Deep SVDD, or one-class learning.
5. **Predict** — for a new song, it measures the distance from its embedding to your taste center. Close enough = **Liked**, too far = **Disliked**.

The Liked/Disliked threshold is set automatically after training, from the 90th percentile of distances across your own liked songs. Training only updates the GRU and a small projection head; the PANN backbone stays frozen throughout.

## 📁 Project layout

```
main.py                   # menu to run everything
app.py                    # Streamlit UI for the same three actions
src/
  data_scraping.py        # downloads audio from YouTube
  audio_preprocessing.py  # loads and cleans up audio files
  hybrid_model.py         # the PANN + GRU model
  train.py                # trains on your liked songs
  predict.py              # checks a new song against your taste
  save_model.py           # saves / loads the trained model
  set_logger.py           # logging setup
model/                    # trained weights saved here
songs/                    # downloaded training songs go here
```

## ⚙️ Setup

```bash
uv sync
```

1. Install `ffmpeg` on your system (needed to extract audio).
2. Download the PANN checkpoint `Cnn14_emb512_mAP=0.420.pth` and place it in `model/`.

## ▶️ Usage

```bash
uv run main.py
```

For first-time use, run the actions in this order:

1. **Scrape playlist** — downloads songs from a playlist you like into `songs/`.
2. **Train model** — trains on `songs/` and saves to `model/pann_gru.pth`.
3. **Get verdict from a video URL** — downloads one song and tells you if you'd like it.

Prefer a UI? The same three actions are available in Streamlit:

```bash
uv run streamlit run app.py
```

> The bundled songs are the example taste this model was trained on — swap them for your own playlist.
