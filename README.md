# Image Segmentation with Grounded SAM

Minimal starter repo to try Grounded SAM via Autodistill on Windows.

## Quickstart

1) Install Conda (Miniconda/Anaconda).
2) Create the environment from `environment.yml`:

```powershell
conda env create -f environment.yml
conda activate grounded-sam
```

3) Optional: install PyTorch with GPU (NVIDIA) support. If you have a compatible CUDA GPU and drivers, install the CUDA build from the official channels:

```powershell
# Optional GPU setup
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

If you do not have a GPU, you can skip this step. Autodistill and Grounded SAM will run on CPU but will be slower.

## Minimal usage

The Grounded SAM teacher is packaged as an Autodistill integration. Example:

```python
# example.py
from autodistill_grounded_sam import GroundedSAM

# Define an ontology (map of class name -> prompt). Adjust to your classes.
ontology = {
    "person": "person",
    "car": "car"
}

model = GroundedSAM(ontology=ontology)

# Predict on a single image
pred = model.predict("path/to/image.jpg")
print(pred)
```

For larger labeling tasks, see the Autodistill docs on creating datasets and running teachers over folders.

## Notes

- The provided `environment.yml` keeps PyTorch out by default so you can choose CPU or GPU builds explicitly.
- If you run into build issues on Windows, ensure you created the env from `conda-forge` and installed PyTorch from the `pytorch`/`nvidia` channels when using CUDA.
- Data directories like `data/`, `datasets/`, `outputs/` are git-ignored by default; customize `.gitignore` as needed.
