# FACES Deployment Notes

## Recommended public demo

Use Hugging Face Spaces with the Gradio SDK.

Required files:

- `app.py`
- `faces_inference/`
- `requirements.txt`
- `logo.png`
- `Binary classification model/best_auc_model_seed64.pth`
- `three-category superclass classification model/best_auc_model_seed53.pth`
- `11-class ResNet50 subtype model/best.pth`

## Hugging Face setup

1. Create a new Space.
2. Choose `Gradio` as the SDK.
3. Push this repository to the Space.
4. Confirm that the Space reads the YAML metadata at the top of `README.md`.
5. Test the Space with a non-face image first, then with a consented frontal face image.

If checkpoint upload fails because files are large, use Git LFS:

```bash
git lfs install
git lfs track "*.pth"
git add .gitattributes
```

## GitHub Pages setup

Use GitHub Pages only for static documentation:

1. Push this repository to GitHub.
2. Open repository `Settings`.
3. Open `Pages`.
4. Select `Deploy from a branch`.
5. Select the target branch and `/docs`.
6. Save.

## RShiny option

RShiny is possible but not recommended for v1. The model stack is Python/PyTorch, so Shiny would need either:

- `reticulate` calling `faces_inference.predict_faces()`, or
- a Shiny frontend calling a Python API.

Both options add deployment complexity compared with the existing Gradio app.
