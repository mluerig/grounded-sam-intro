# Image Segmentation with Grounded SAM

This is a minimal starter repo to use the Grounded Segment Anything Model (GroundedSAM) for image segmentation via text-based input.

- [GroundedSAM repo](https://github.com/IDEA-Research/Grounded-Segment-Anything)
- [Autodistill repo](https://github.com/autodistill/autodistill-grounded-sam)

![SAM3 Animation](sam3-animation.gif)
(Note: this animation shows the SAM3 web interface, but the principle is the same)

## Quickstart

1\. Clone the repo.

Make sure you have *git* and *git lfs* installed on your machine (e.g. via `git lfs install`).

```powershell
git clone https://github.com/mluerig/demo-grounded-sam
```

2\. Install *mamba* via [miniforge](https://github.com/conda-forge/miniforge), or *conda* (Miniconda/Anaconda).  

3\. Open a terminal at the repo root dir, and create the environment from `environment.yml`:

```powershell
mamba env create -f environment.yml -n grounded-sam1
mamba activate grounded-sam1
```

4\. Optional: install PyTorch with GPU (NVIDIA) support. If you have a compatible CUDA GPU and drivers, install the CUDA build from the official channels:

```powershell
mamba install nvidia::cuda-toolkit==12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --force-reinstall
```

If you do not have a GPU, you can skip this step. Autodistill and Grounded SAM will run on CPU but will be MUCH slower.

5\. Unzip the example data (*data_raw\input_imgs\butterflies.zip*), and run the notebook 

## Notes

- The provided `environment.yml` keeps PyTorch out by default so you can choose CPU or GPU builds explicitly.
- The model will likely produce some false detections/segmentations, which is expected.

