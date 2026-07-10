# CDA — Contract Document Analysis (WIP)

An early-stage exploration into LLM-based contract clause retrieval and question answering over the [CUAD](https://www.atticusprojectai.org/cuad) (Contract Understanding Atticus Dataset) contract corpus, using the Google Gen AI SDK.

## Repository structure
* `data/`: local copy of the CUAD dataset (`CUADv1.json`, `train_separate_questions.json`, `test.json`).
* `src/`: skeleton modules for embeddings, retrieval models, and querying — implementation in progress.
* `main.py`: entry point (not yet implemented).
* `visualise_data.ipynb`: exploratory notebook over the CUAD dataset.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/) and are self-contained within this project folder:
```bash
uv sync
```

This project also expects a `GEMINI_API_KEY` (or equivalent Google Gen AI credentials) to be available in your environment to call the API.

This project is a work in progress — most modules are currently empty stubs.
