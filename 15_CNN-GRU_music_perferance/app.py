import streamlit as st
from pathlib import Path
from src import (
    audio_scraper,
    clear_directory,
    make_dataloader,
    load_model,
    predict_preferences,
    train,
    PannGru,
)

songs_dir = Path(__file__).resolve().parent / "songs"
pred_dir = songs_dir / "pred"

st.title("Music Taste Predictor")

tab1, tab2, tab3 = st.tabs(["Predict", "Scrape playlist", "Train"])

with tab1:
    url = st.text_input("YouTube video URL")
    if st.button("Get verdict"):
        pred_dir.mkdir(parents=True, exist_ok=True)
        clear_directory(pred_dir)
        audio_scraper(url, pred_dir)

        dataloader = make_dataloader(pred_dir)
        model = load_model(PannGru())

        st.write(predict_preferences(model, dataloader))

with tab2:
    playlist_url = st.text_input("YouTube playlist URL", key="playlist")
    if st.button("Scrape playlist"):
        audio_scraper(playlist_url, songs_dir)
        st.success("Done scraping playlist.")

with tab3:
    if st.button("Train model"):
        dataloader = make_dataloader(songs_dir)
        train(PannGru(), dataloader)
        st.success("Training complete.")
