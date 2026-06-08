# Script Guide

This guide explains what each script does.

It also lists the main settings to edit.

Most settings are constants near the top of each script.

Some settings can also be passed as command-line arguments.

## General Notes

Run commands from the repository root.

Example:

```bash
python scripts/train_pitch_model.py
```

The scripts build many paths relative to the `scripts/` folder.

So generated data often appears inside `scripts/`, not beside the root README.

Do not rename generated folders unless you also update the script settings or command arguments.

## Dataset Collection

Script:

```text
scripts/collect_pitch_dataset.py
```

Purpose:

- Connects to CARLA.
- Loads `Town05`.
- Spawns a vehicle.
- Adds a semantic segmentation camera.
- Saves road and road-line mask images.
- Saves pitch labels and metadata to CSV files.

Run:

```bash
python scripts/collect_pitch_dataset.py
```

CARLA must already be running.

Default output:

```text
scripts/dataset_pitch/
```

Important output files:

```text
scripts/dataset_pitch/train.csv
scripts/dataset_pitch/val.csv
scripts/dataset_pitch/dataset_all.csv
scripts/dataset_pitch/train/masks/
scripts/dataset_pitch/val/masks/
```

Important settings near the top of the file:

- `MAP_NAME = "Town05"`
- `CAMERA_WIDTH = 1280`
- `CAMERA_HEIGHT = 720`
- `CAMERA_FOV = 90`
- `SENSOR_TICK = 0.05`
- `SIM_DURATION_SECONDS = 300`
- `DISPLAY_MASK = True`
- `TRAIN_RATIO = 0.80`
- `SPLIT_MODE = "random"`
- `FORCE_BEST_TEST_SPAWN = True`

Pay attention to:

- `DISPLAY_MASK = True` opens an OpenCV window.
- Press `q` in the mask window to quit.
- Press `Ctrl+C` in the terminal to stop.
- `SPLIT_MODE = "random"` may place very similar nearby frames in both train and validation.
- `SPLIT_MODE = "block"` is better if you want less frame leakage.
- Existing files in `scripts/dataset_pitch/` may be overwritten.

## Model Training

Script:

```text
scripts/train_pitch_model.py
```

Purpose:

- Reads the dataset CSV files.
- Loads the saved mask images.
- Trains a MobileNetV3-small regression model.
- Saves model checkpoints, plots, and validation CSV files.

Run:

```bash
python scripts/train_pitch_model.py
```

Default input:

```text
scripts/dataset_pitch/
```

Default outputs:

```text
scripts/training_output/
scripts/models/
```

Important output files:

```text
scripts/models/mobilenet_pitch_best.pth
scripts/models/mobilenet_pitch_last.pth
scripts/training_output/train_history.csv
scripts/training_output/loss_curve.png
scripts/training_output/val_prediction_vs_true_last.png
```

Useful command-line options:

```bash
python scripts/train_pitch_model.py --epochs 10
python scripts/train_pitch_model.py --batch-size 16
python scripts/train_pitch_model.py --dataset-dir scripts/dataset_pitch
python scripts/train_pitch_model.py --model-dir models
```

Important settings near the top of the file:

- `DEFAULT_DATASET_DIR`
- `DEFAULT_OUTPUT_DIR`
- `DEFAULT_MODEL_DIR`
- `DEFAULT_TARGET_COLUMN = "true_vehicle_pitch_deg"`
- `IMAGE_SIZE = 224`
- `BATCH_SIZE = 32`
- `NUM_EPOCHS = 30`
- `LEARNING_RATE = 1e-3`
- `NUM_WORKERS = 4`

Pay attention to:

- The dataset must exist before training.
- The CSV files must contain `mask_path`.
- The default target is `true_vehicle_pitch_deg`.
- You can switch to `waypoint_pitch_deg`, but then results mean something different.
- If training crashes with multiprocessing issues, try `--num-workers 0`.
- If CUDA memory is too low, reduce `--batch-size`.

## Final Evaluation

Script:

```text
scripts/run_pitch_kf_final_clean.py
```

Purpose:

- Connects to CARLA.
- Loads the trained MobileNet pitch model.
- Uses the semantic camera as the learned pitch measurement.
- Uses IMU gyroscope data for Kalman Filter prediction.
- Computes a GNSS pitch baseline.
- Saves CSV logs, plots, and error summaries.

Run with the included root model:

```bash
python scripts/run_pitch_kf_final_clean.py --model-path models/mobilenet_pitch_best.pth
```

Run with a freshly trained default model:

```bash
python scripts/run_pitch_kf_final_clean.py
```

CARLA must already be running.

Default output:

```text
scripts/sensor_fusion_ml_camera_run/
```

Important output files:

```text
scripts/sensor_fusion_ml_camera_run/sensor_log_clean.csv
scripts/sensor_fusion_ml_camera_run/error_summary.csv
scripts/sensor_fusion_ml_camera_run/camera_pitch_comparison.png
scripts/sensor_fusion_ml_camera_run/gnss_pitch_comparison.png
scripts/sensor_fusion_ml_camera_run/kf_pitch_comparison.png
scripts/sensor_fusion_ml_camera_run/all_sensors_pitch_comparison.png
```

Useful command-line options:

```bash
python scripts/run_pitch_kf_final_clean.py --model-path models/mobilenet_pitch_best.pth
python scripts/run_pitch_kf_final_clean.py --duration 60
python scripts/run_pitch_kf_final_clean.py --no-display
python scripts/run_pitch_kf_final_clean.py --save-masks
python scripts/run_pitch_kf_final_clean.py --full-csv
```

Important settings near the top of the file:

- `MAP_NAME = "Town05"`
- `OUTPUT_DIR_NAME = "sensor_fusion_ml_camera_run"`
- `DEFAULT_MODEL_PATH`
- `CAMERA_WIDTH = 1280`
- `CAMERA_HEIGHT = 720`
- `CAMERA_FOV = 90`
- `CAMERA_SENSOR_TICK = 0.05`
- `IMU_SENSOR_TICK = 0.05`
- `GNSS_SENSOR_TICK = 0.05`
- `SIM_DURATION_SECONDS = 170`
- `DISPLAY_MASK = True`
- `SAVE_MASK_IMAGES = False`
- `SAVE_PLOTS_DURING_RUN = True`
- `FORCE_BEST_TEST_SPAWN = True`

Pay attention to:

- The included model is in `models/`, not `scripts/models/`.
- Pass `--model-path models/mobilenet_pitch_best.pth` when using the included model.
- The default trained model path is `scripts/models/mobilenet_pitch_best.pth`.
- You can also put the best model at `scripts/models/mobilenet_pitch_best.pth` and run the final script without `--model-path`.
- `--no-display` is useful if OpenCV windows cause problems.
- `--save-masks` can create many image files.
- The GNSS pitch estimate may be unavailable when horizontal movement is too small.
- Existing files in the run output folder may be overwritten.

## Editing Settings Safely

Change one setting at a time.

Run a short test after each change.

For quick tests, use a shorter duration:

```bash
python scripts/run_pitch_kf_final_clean.py --model-path models/mobilenet_pitch_best.pth --duration 30
```

Keep camera resolution and model image size separate.

Camera resolution controls CARLA images.

Model image size controls the neural network input.

Do not change semantic tag values unless you know the CARLA version changed them.

For CARLA 0.9.16:

- Road is `1`.
- Road line is `24`.

If you change output folders, update the next step too.

Example:

- If dataset output changes, pass the new folder to training with `--dataset-dir`.
- If model output changes, pass the new checkpoint to evaluation with `--model-path`.
