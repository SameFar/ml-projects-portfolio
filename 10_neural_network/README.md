<div align="center">

# 👗 Fashion Recommendation Neural Network

**A multi-head net, built twice — from scratch in NumPy and in PyTorch.**

![Deep Learning](https://img.shields.io/badge/🟣_DEEP_LEARNING-8957e5?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Vercel](https://img.shields.io/badge/Vercel-000000?style=flat-square&logo=vercel&logoColor=white)

</div>

---

Give it someone's hair colour, eye colour, skin tone, undertone, and torso length, and it predicts a set of fashion recommendations — colors to wear and avoid, materials, patterns, jewelry metal, and color-wheel region. It's a multi-head network: one shared hidden layer feeding six output heads, one per category.

The folder holds **two parallel implementations** of the same model — one from scratch in NumPy, one in PyTorch — so there's a hand-written version alongside one deployable without dragging PyTorch into a serverless function.

## 🧠 How it works

Input is one-hot encoded from five categorical features, through a shared ReLU hidden layer (He/Kaiming init), branching into six heads (Xavier/Glorot init), each a softmax trained with categorical cross-entropy (every head is a pick-one-of-N choice).

- `src/neural_network.py` — both models: a `NeuralNetwork` with hand-written forward/backward passes and manual gradient descent, and a `MultiHeadNeuralNetwork` PyTorch `nn.Module` with the same architecture on `autograd`.
- `src/train.py` — trains the NumPy model (3000 epochs), saving to `src/model/fashion_nn_weights.pkl` and `api/model/fashion_nn_weights.pkl`.
- `src/torch_train.py` — trains the PyTorch model (2000 epochs, SGD), saving to `src/model/torch_model.pth`.
- Both use the same decayed LR schedule (0.05, decaying 0.00001/step down to 0.001).
- `data/recommendations.csv` — training data ([Kaggle](https://www.kaggle.com/datasets/suryaprabha19/fashion-and-color-recommendation-dataset)), one-hot encoded in `src/data_preprocessing.py`. EDA lives in `data_visualisation.ipynb`.

Two ways to serve, both behind `POST /predict` and the same `FashionInput` schema:

- `api/app.py` — loads the NumPy model; deployed to Vercel as a serverless function (`api/vercel.json` routes all traffic to it; `api/requirements.txt` exists only because Vercel's Python builder reads it directly and doesn't know about uv).
- `app.py` — loads the PyTorch model from `src/model/torch_model.pth`, for running locally on port 8001.

## 🚀 Getting started

```bash
uv sync

# train the from-scratch NumPy model
uv run -m src.train

# train the PyTorch model
uv run -m src.torch_train

# run the PyTorch inference API locally
uv run app.py

# run the NumPy inference API locally (the one Vercel deploys)
cd api && uv run uvicorn app:app --reload --port 8000
```
