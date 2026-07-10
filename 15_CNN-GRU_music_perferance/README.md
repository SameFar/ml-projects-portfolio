# CNN-GRU Music Preference Model

This project tries to guess if you would like a song, just by listening to it.

You give it a YouTube playlist of songs you like. It downloads them, listens to
them with a pretrained audio model, and learns what your "taste" sounds like.
After that, you can give it a link to any YouTube video and it will tell you
"Liked" or "Disliked" based on how close the song sounds to your taste.

## How it works

1. **Download songs** - `yt-dlp` grabs the audio from a YouTube playlist and
   saves each track as a 30-second mp3 clip (mono, 16kHz).
2. **Listen to the songs** - Each song is cut into 5-second chunks. A
   pretrained model called PANN (Cnn14) turns every chunk into a list of
   numbers that describes what the audio sounds like. This part is frozen and
   never trained.
3. **Learn the pattern over time** - The chunk descriptions for a song are fed
   in order into a GRU (a type of recurrent neural network), so the model
   picks up how the song changes over time, not just a single snapshot.
4. **Learn your taste** - Instead of teaching the model "like" vs "dislike",
   it only ever sees songs you like. It learns a single point in space (the
   "center") that represents your average taste, and pulls all your liked
   songs close to that point. This is called Deep SVDD, or one-class
   learning.
5. **Predict** - For a new song, the model checks how far its embedding is
   from your taste center. Close enough = "Liked", too far = "Disliked".

## Project layout

```
main.py                  # simple menu to run everything
app.py                    # Streamlit UI version of the same three actions
src/
  data_scraping.py        # downloads audio from YouTube
  audio_preprocessing.py  # loads and cleans up audio files
  hybrid_model.py          # the PANN + GRU model
  train.py                 # trains the model on your liked songs
  predict.py                # checks a new song against your taste
  save_model.py              # saves / loads the trained model
  set_logger.py                # logging setup
model/                    # trained model weights get saved here
songs/                    # downloaded training songs go here
```

## Setup

1. Dependencies are managed with [uv](https://docs.astral.sh/uv/) and are self-contained within this project folder:
   ```bash
   uv sync
   ```
2. Make sure `ffmpeg` is installed on your system (needed to extract audio).
3. Download the PANN checkpoint `Cnn14_emb512_mAP=0.420.pth` and place it in
   the `model/` folder.

## Usage

Run the program:

```bash
uv run main.py
```

You'll see a menu:

- **Option 2 - Scrape playlist**: Downloads songs from a YouTube playlist you
  like into the `songs/` folder. Do this first.
- **Option 3 - Train model**: Trains the model on the songs in `songs/` and
  saves it to `model/pann_gru.pth`.
- **Option 1 - Get verdict from a video URL**: Downloads one song from a
  YouTube link and tells you if the trained model thinks you'd like it.
- **Option 4 - Exit**.

The comment in the menu says it best: for first-time use, run option 2, then
option 3, before trying option 1.

There's also a Streamlit version of the same three actions (predict, scrape
playlist, train) if you'd rather click through a UI:

```bash
uv run streamlit run app.py
```

## Notes

- The songs in this project are the songs used to train this specific model,
  they're a private/example music taste and can be swapped for your own
  playlist.
- Training only updates the GRU and the small projection head on top, the
  PANN backbone stays frozen the whole time.
- The "threshold" that decides Liked vs Disliked is calculated automatically
  after training, based on the 90th percentile of distances in your own
  liked songs.
