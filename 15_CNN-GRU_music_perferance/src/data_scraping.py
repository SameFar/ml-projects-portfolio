import re
import logging
from pathlib import Path

import yt_dlp


def sanitize_filename(name: str) -> str:
    """Replicates standard yt-dlp filename cleaning to ensure accurate cache hits."""
    return re.sub(r'[\\/*?:"<>|]', "_", name)


def clear_directory(dir_path: Path):
    for item in dir_path.iterdir():
        item.unlink()  # Delete file


def audio_scraper(url: str, songs_dir: Path):
    songs_dir.mkdir(parents=True, exist_ok=True)

    meta_opts = {
        "extract_flat": True,
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
    }

    # Extract metadata (works for both playlists and single videos)
    try:
        with yt_dlp.YoutubeDL(meta_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        logging.error(f"Failed to extract metadata: {e}")
        return

    # Determine whether this is a playlist or a single video
    if "entries" in info:
        video_entries = [entry for entry in info["entries"] if entry]
        logging.info(f"Found {len(video_entries)} tracks in the playlist.")
    else:
        video_entries = [info]
        logging.info("Processing a single video.")

    for index, entry in enumerate(video_entries):
        if not entry:
            continue

        video_id = entry.get("id")
        if not video_id:
            logging.warning(f"Skipping entry {index + 1}: Missing video ID.")
            continue

        video_url = (
            entry.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}"
        )

        raw_title = entry.get("title", f"track_{index + 1}")
        video_title = sanitize_filename(raw_title)

        expected_file_path = songs_dir / f"{video_title}.mp3"

        # Skip if already downloaded
        if expected_file_path.exists():
            logging.info(
                f"[{index + 1}/{len(video_entries)}] Skipping '{video_title}' (Already cached)"
            )
            continue

        logging.info(f"[{index + 1}/{len(video_entries)}] Processing: {video_title}")

        output_template = str(songs_dir / f"{video_title}.%(ext)s")

        dl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "quiet": True,
            "no_warnings": True,
            "ignoreerrors": True,
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "postprocessor_args": [
                "-ar",
                "16000",
                "-ac",
                "1",
            ],
            "external_downloader": "ffmpeg",
            "external_downloader_args": {
                "ffmpeg_i": [
                    "-ss",
                    "00:00:45",
                    "-t",
                    "00:00:30",
                ]
            },
        }

        try:
            with yt_dlp.YoutubeDL(dl_opts) as ydl:
                ydl.download([video_url])

            if expected_file_path.exists():
                logging.info(f"Successfully added: {video_title}")
            else:
                logging.error(f"Failed to generate output for: {video_title}")

        except Exception as e:
            logging.error(f"Skipping '{video_title}' due to unexpected error: {e}")

    logging.info("Dataset generation completed successfully!")
