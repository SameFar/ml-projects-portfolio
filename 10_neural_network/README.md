# Fashion Recommendation Neural Network

Give it someone's hair colour, eye colour, skin tone, undertone, and torso length, and it predicts a set of fashion recommendations for them — clothing colors to wear and avoid, materials, patterns, jewelry metal, and color-wheel region. It's a multi-head neural network: one shared hidden layer feeding into six separate output heads, one per recommendation category.

This folder has two parallel implementations of the same model: one written from scratch in NumPy, and one in PyTorch. I built both to have a from-scratch version alongside a version I could actually deploy without dragging PyTorch into a serverless function.

## How it works

The input is one-hot encoded from five categorical features. It goes through a single shared hidden layer with a ReLU activation (He/Kaiming-initialized), which then branches into six output heads (Xavier/Glorot-initialized), each ending in a softmax and trained with categorical cross-entropy, since every head is a pick-one-of-N choice.

- `src/neural_network.py` has both versions of the model: a `NeuralNetwork` class with hand-written forward/backward passes and manual gradient descent, and a `MultiHeadNeuralNetwork` PyTorch `nn.Module` with the same architecture built on `autograd`.
- `src/train.py` trains the NumPy model for 3000 epochs and saves the weights to both `src/model/fashion_nn_weights.pkl` and `api/model/fashion_nn_weights.pkl`.
- `src/torch_train.py` trains the PyTorch model for 2000 epochs with SGD, and saves it to `src/model/torch_model.pth`.
- Both training loops use the same manually-decayed learning rate schedule (starts at 0.05, decays by 0.00001 per step down to a floor of 0.001).
- `data/recommendations.csv` is the training data ([Kaggle: fashion-and-color-recommendation-dataset](https://www.kaggle.com/datasets/suryaprabha19/fashion-and-color-recommendation-dataset)), loaded and one-hot encoded in `src/data_preprocessing.py`.
- `data_visualisation.ipynb` has the exploratory data analysis.

There are two ways to serve predictions, both behind the same `POST /predict` endpoint and `FashionInput` pydantic schema:

- `api/app.py` — loads the NumPy model from `api/model/fashion_nn_weights.pkl`. This is the one deployed to Vercel as a serverless function (`api/vercel.json` routes all traffic to it); `api/requirements.txt` exists solely because Vercel's Python builder reads it directly and doesn't know about uv.
- `src/app.py` — loads the PyTorch model from `src/model/torch_model.pth`, for running the same thing locally on port 8001.

## Getting started

```bash
uv sync

# train the from-scratch NumPy model
uv run src/train.py

# train the PyTorch model
uv run src/torch_train.py

# run the PyTorch inference API locally
uv run src/app.py

# run the NumPy inference API locally (the one Vercel deploys)
cd api && uv run uvicorn app:app --reload --port 8000
```
