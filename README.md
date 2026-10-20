# Plant Disease Detector

A Streamlit image-classification demo that predicts one of the plant leaf
classes represented by the bundled Keras model and displays a translated
recommendation in English, Telugu, or Hindi.

## Status

This is a functional demo repository, not a validated agricultural diagnostic
system. The repository does not include an evaluation report, test suite, or
deployment configuration. Predictions and confidence values should be treated
as model outputs for experimentation, not professional advice.

## Features

- Upload JPG, JPEG, or PNG leaf images.
- Run a TensorFlow/Keras model on a 128×128 RGB image.
- Display the predicted class and model confidence.
- Show a simple severity label and a language-specific recommendation.
- Choose English, Telugu, or Hindi UI text.

## Technology stack

- Python
- Streamlit
- TensorFlow/Keras
- Pillow and NumPy
- `streamlit-lottie` for the UI animation

## Model artifact and provenance

The application currently loads `trained_plant_disease_model.keras` from the
project root, so the file is required for the existing inference flow. The
tracked artifact is 94,230,432 bytes (approximately 89.9 MiB), below GitHub's
100 MB individual-file limit but close enough that Git LFS or a release asset
would be preferable for future model revisions.

The file metadata identifies a TensorFlow/Keras 2.10.0 model with 38 output
classes. This repository does not record the training dataset, model author,
source license, or redistribution permission. Do not claim dataset provenance or
publish another copy until those rights are confirmed.

## Installation and usage

Use a Python environment compatible with the pinned TensorFlow dependency:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run main.py
```

The application expects `trained_plant_disease_model.keras` in the project root.
The model is currently tracked as a large binary artifact; review whether it
should remain in Git or move to a release asset/LFS before future releases.

## Project structure

```text
main.py                              Streamlit UI and inference flow
trained_plant_disease_model.keras    Bundled TensorFlow/Keras model
plant.json                           Local animation/config asset
requirements.txt                     Runtime dependencies
.devcontainer/                       Development-container metadata
```

## Architecture

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the input, inference,
recommendation, and limitation boundaries.

## Testing

No automated tests are currently provided. The available low-risk syntax check
is:

```bash
python -m py_compile main.py
```

This does not load TensorFlow or verify prediction quality.

## Limitations

- The class list and recommendations are embedded in `main.py`.
- There is no persisted user/session data layer.
- No calibration, held-out evaluation, confusion matrix, or dataset card is
  included in this repository.
- Recommendations are informational and should be checked with a qualified
  local agricultural expert.
- No production deployment is configured in the repository.

## License

No license is currently supplied. Do not assume permission to redistribute the
model, source, or any upstream training data until the licensing status is
confirmed.
