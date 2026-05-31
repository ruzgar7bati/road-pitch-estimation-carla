# Road Pitch Estimation in CARLA

Road pitch estimation and sensor fusion using semantic camera measurements, MobileNetV3 regression, IMU data, GNSS baseline estimation, and Kalman filtering.

**EE4084 Project**
Department of Electrical and Electronics Engineering
Marmara University

**Authors**

* Rüzgar Batı Okay
* Alperen Tufan Pelit

## Overview

This project estimates road pitch in the CARLA simulator using a semantic segmentation camera and sensor fusion.

The pipeline consists of:

* Semantic camera -> road and road-line masks
* MobileNetV3 -> camera pitch estimation
* IMU gyroscope -> Kalman Filter prediction
* Camera pitch -> Kalman Filter correction
* GNSS -> baseline comparison
* CARLA ground truth and waypoint pitch -> evaluation

## Software Versions

| Component   | Version           |
| ----------- | ----------------- |
| CARLA       | 0.9.16            |
| Python      | 3.11              |
| PyTorch     | CUDA build        |
| CUDA        | 13.2              |
| OpenCV      | Latest compatible |
| torchvision | Latest compatible |
| matplotlib  | Latest compatible |

## Dependencies

* CARLA: https://carla.org/
* PyTorch: https://pytorch.org/
* torchvision: https://pytorch.org/vision/
* OpenCV: https://opencv.org/
* NumPy: https://numpy.org/
* Matplotlib: https://matplotlib.org/

## Project Structure

```text
scripts/
├── 01_collect_pitch_dataset.py
├── 02_train_pitch_model.py
└── 05_run_kf_evaluation.py

models/
└── mobilenet_pitch_best.pth

paper/
└── final_paper.pdf

example_outputs/
├── plots
├── csv_logs
└── error_summary
```

## Setup

### 1. Install CARLA 0.9.16

Download and run CARLA:

https://github.com/carla-simulator/carla/releases

### 2. Create Python Environment

```bash
python -m venv venv
```

Activate the environment.

### 3. Install Dependencies

```bash
pip install torch torchvision opencv-python numpy matplotlib pillow
```

Install the CARLA Python API compatible with CARLA 0.9.16.

## Workflow

### Dataset Collection

```bash
python scripts/collect_pitch_dataset.py
```

Generates semantic road masks and pitch labels.

### Train MobileNet Model

```bash
python scripts/train_pitch_model.py
```

Trains the MobileNetV3 pitch regression model.

### Run Sensor Fusion Evaluation

```bash
python scripts/run_kf_evaluation.py
```

Runs live pitch estimation using:

* Semantic camera
* IMU gyroscope
* GNSS baseline
* Kalman Filter fusion

## Outputs

The project generates:

* Trained MobileNet model
* CSV sensor logs
* Error summaries
* Camera/KF/GNSS comparison plots
* Semantic mask examples

## References

1. E. Ustunel and E. Masazade, *Iterative Range and Road Parameters Estimation Using Monocular Camera on Highways*, IEEE T-ITS, 2023.

2. A. Dosovitskiy et al., *CARLA: An Open Urban Driving Simulator*, CoRL, 2017.

3. A. Howard et al., *Searching for MobileNetV3*, ICCV, 2019.

4. A. Paszke et al., *PyTorch: An Imperative Style, High-Performance Deep Learning Library*, NeurIPS, 2019.

## License

MIT License.
