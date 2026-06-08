# Setup Guide

This guide explains how to set up the project files, CARLA, Python, and the run folders.

The short version is:

```bash
python -m venv venv
pip install -r requirements.txt
python scripts/collect_pitch_dataset.py
python scripts/train_pitch_model.py
python scripts/run_pitch_kf_final_clean.py --model-path models/mobilenet_pitch_best.pth
```

CARLA must be running before the first and third script.

## 1. Folder Layout

Keep the repository folder structure as it is.

Important folders:

```text
scripts/
+-- collect_pitch_dataset.py
+-- train_pitch_model.py
+-- run_pitch_kf_final_clean.py

models/
+-- mobilenet_pitch_best.pth

example_outputs/
+-- example plots from a completed run

Papers/
+-- paper, proposal, and presentation files

docs/
+-- setup and script notes
```

The scripts use their own location to build output paths.

This means many generated files are written inside `scripts/`.

Do not assume outputs always go to the repository root.

## 2. Install CARLA

Install CARLA 0.9.16.

Download it from:

https://github.com/carla-simulator/carla/releases

After downloading, extract CARLA to a stable location.

Example on Windows:

```text
C:\CARLA_0.9.16
```

Start CARLA before running data collection or final evaluation.

On Windows this is usually:

```text
CarlaUE4.exe
```

The scripts connect to:

```text
127.0.0.1:2000
```

If CARLA is not running, those scripts will fail.

## 3. Install the CARLA Python API

The Python package must match both CARLA and Python.

This project uses:

- CARLA 0.9.16
- Python 3.11
- Windows wheel in the original development setup

The `requirements.txt` file currently contains a local CARLA wheel path.

It looks like this:

```text
carla @ file:///C:/CARLA_0.9.16/PythonAPI/carla/dist/...
```

That path only works if your CARLA folder is in the same place.

If your CARLA folder is somewhere else, edit the path in `requirements.txt`.

You can also install the wheel manually:

```bash
pip install C:\path\to\CARLA_0.9.16\PythonAPI\carla\dist\carla-0.9.16-cp311-cp311-win_amd64.whl
```

Use the wheel that matches your Python version.

For this repo, that should be a `cp311` wheel.

## 4. Create the Python Environment

Create a virtual environment from the repository root:

```bash
python -m venv venv
```

Activate it.

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

On Command Prompt:

```bat
venv\Scripts\activate.bat
```

Then install dependencies:

```bash
pip install -r requirements.txt
```

If this fails on the CARLA line, fix the CARLA wheel path first.

If it fails on PyTorch or torchvision, install a CUDA-compatible build from:

https://pytorch.org/

## 5. Run Order

Run the scripts in this order.

Start CARLA first.

```bash
python scripts/collect_pitch_dataset.py
python scripts/train_pitch_model.py
python scripts/run_pitch_kf_final_clean.py --model-path models/mobilenet_pitch_best.pth
```

The included model is in:

```text
models/mobilenet_pitch_best.pth
```

The final evaluation script normally looks under `scripts/models/`.

That is why the command above passes `--model-path`.

Another option is to copy or move the best model file into:

```text
scripts/models/mobilenet_pitch_best.pth
```

If the file is there, the final script can find it with its default settings.

If you train a new model, the training script writes it to:

```text
scripts/models/mobilenet_pitch_best.pth
```

Then you can run the final script without `--model-path`, or you can pass the new path explicitly.

## 6. Generated Files

Dataset collection writes:

```text
scripts/dataset_pitch/
+-- train.csv
+-- val.csv
+-- dataset_all.csv
+-- train/masks/
+-- val/masks/
```

Training writes:

```text
scripts/training_output/
+-- train_history.csv
+-- val_predictions_best.csv
+-- val_predictions_last.csv
+-- loss_curve.png
+-- val_prediction_vs_true_last.png
+-- val_error_histogram_last.png
```

Training also writes model checkpoints:

```text
scripts/models/
+-- mobilenet_pitch_best.pth
+-- mobilenet_pitch_last.pth
```

Final evaluation writes:

```text
scripts/sensor_fusion_ml_camera_run/
+-- sensor_log_clean.csv
+-- error_summary.csv
+-- camera_pitch_comparison.png
+-- gnss_pitch_comparison.png
+-- kf_pitch_comparison.png
+-- all_sensors_pitch_comparison.png
```

If `--save-masks` is used, it also writes mask images under the run folder.

## 7. Common Things to Check

Check that CARLA is open before running collection or evaluation.

Check that the map can load as `Town05`.

Check that Python is 3.11.

Check that the CARLA wheel path in `requirements.txt` is correct.

Check that the model path exists before final evaluation.

Check that your GPU/CUDA PyTorch install works if you want CUDA acceleration.

The scripts can still run on CPU, but training will be slower.

## 8. More Script Details

See `docs/scripts.md` for how each script works, what settings matter, and what to edit carefully.
