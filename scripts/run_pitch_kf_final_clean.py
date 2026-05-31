import carla
import random
import time
import csv
import os
import math
import argparse

import numpy as np
import cv2

import torch
import torch.nn as nn
from PIL import Image

try:
    import torchvision
    from torchvision import transforms
except ImportError as exc:
    raise ImportError(
        "torchvision is required. Install it with:\n"
        "pip install torchvision"
    ) from exc

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# ============================================================
# Purpose
# ============================================================
#
# Main CARLA experiment script.
#
# Pipeline:
#   semantic camera -> road/line mask -> MobileNet pitch
#   IMU gyro_y      -> Kalman prediction
#   camera pitch    -> Kalman correction
#   GNSS pitch      -> baseline only
#
# References:
#   true vehicle pitch and waypoint pitch are used only for evaluation.
#


# ============================================================
# Settings
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MAP_NAME = "Town05"

OUTPUT_DIR_NAME = "sensor_fusion_ml_camera_run"
OUTPUT_DIR = os.path.join(SCRIPT_DIR, OUTPUT_DIR_NAME)

CSV_NAME = "sensor_log_clean.csv"
SUMMARY_CSV_NAME = "error_summary.csv"

CAMERA_PLOT_NAME = "camera_pitch_comparison.png"
GNSS_PLOT_NAME = "gnss_pitch_comparison.png"
KF_PLOT_NAME = "kf_pitch_comparison.png"
ALL_SENSOR_PLOT_NAME = "all_sensors_pitch_comparison.png"

DEFAULT_MODEL_PATH = os.path.join(SCRIPT_DIR, "models", "mobilenet_pitch_best.pth")

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FOV = 90

# 20 Hz sensors.
CAMERA_SENSOR_TICK = 0.05
IMU_SENSOR_TICK = 0.05
GNSS_SENSOR_TICK = 0.05

SIM_DURATION_SECONDS = 170

DISPLAY_MASK = True

SAVE_MASK_IMAGES = False
MASK_SAVE_EVERY_N_FRAMES = 20

SAVE_PLOTS_DURING_RUN = True
PLOT_SAVE_EVERY_N_FRAMES = 20

FORCE_BEST_TEST_SPAWN = True
SPAWN_LOOKAHEAD_DISTANCES = [10.0, 20.0, 30.0, 40.0, 60.0]

# CARLA 0.9.16 semantic tags.
SEMANTIC_TAG_ROAD = 1
SEMANTIC_TAG_ROAD_LINE = 24

DILATE_ROAD_LINES = True
ROAD_LINE_DILATE_KERNEL_SIZE = 3
ROAD_LINE_DILATE_ITERATIONS = 1

# GNSS pitch baseline.
GNSS_PITCH_MIN_HORIZONTAL_DISTANCE = 0.25
MAX_REASONABLE_GNSS_PITCH_DEG = 20.0

DEFAULT_MODEL_IMAGE_SIZE = 224


# ============================================================
# Latest sensor data
# ============================================================

latest_semantic = {
    "frame": None,
    "timestamp": None,
    "mask_bgr": None,
}

latest_imu = {
    "frame": None,
    "timestamp": None,
    "accel": None,
    "gyro": None,
    "compass": None,
}

latest_gnss = {
    "frame": None,
    "timestamp": None,
    "latitude": None,
    "longitude": None,
    "altitude": None,
}


# ============================================================
# Sensor callbacks
# ============================================================

def semantic_callback(image):
    """Read semantic tags and create the training-compatible mask."""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))

    # CARLA semantic tag is in the red channel of BGRA.
    tag_image = array[:, :, 2]

    latest_semantic["frame"] = image.frame
    latest_semantic["timestamp"] = image.timestamp
    latest_semantic["mask_bgr"] = make_road_roadline_mask_from_tags(tag_image)


def imu_callback(data):
    """Store latest IMU sample."""
    latest_imu["frame"] = data.frame
    latest_imu["timestamp"] = data.timestamp
    latest_imu["accel"] = data.accelerometer
    latest_imu["gyro"] = data.gyroscope
    latest_imu["compass"] = data.compass


def gnss_callback(data):
    """Store latest GNSS sample."""
    latest_gnss["frame"] = data.frame
    latest_gnss["timestamp"] = data.timestamp
    latest_gnss["latitude"] = data.latitude
    latest_gnss["longitude"] = data.longitude
    latest_gnss["altitude"] = data.altitude


# ============================================================
# Basic helpers
# ============================================================

def wrap_angle_180(angle_deg):
    """Wrap angle to [-180, 180)."""
    if angle_deg is None:
        return None
    return ((angle_deg + 180.0) % 360.0) - 180.0


def get_vehicle_speed_mps(vehicle):
    """Return vehicle speed magnitude."""
    velocity = vehicle.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def finite_xy(series):
    """Drop None/NaN points before plotting."""
    xs = []
    ys = []

    for x, y in series:
        if y is None:
            continue
        if isinstance(y, float) and not math.isfinite(y):
            continue
        xs.append(x)
        ys.append(y)

    return xs, ys


def safe_error(estimate, truth):
    """Return estimate - truth if both exist."""
    if estimate is None or truth is None:
        return None
    return estimate - truth


# ============================================================
# Semantic mask
# ============================================================

def make_road_roadline_mask_from_tags(tag_image):
    """
    Same mask format as training:
      B = road
      G = road lines
      R = road + road lines
    """
    road_mask = (tag_image == SEMANTIC_TAG_ROAD).astype(np.uint8) * 255
    roadline_mask = (tag_image == SEMANTIC_TAG_ROAD_LINE).astype(np.uint8) * 255

    # Make thin lane markings easier for the model to see.
    if DILATE_ROAD_LINES:
        kernel = np.ones(
            (ROAD_LINE_DILATE_KERNEL_SIZE, ROAD_LINE_DILATE_KERNEL_SIZE),
            dtype=np.uint8
        )
        roadline_mask = cv2.dilate(
            roadline_mask,
            kernel,
            iterations=ROAD_LINE_DILATE_ITERATIONS
        )

    combined_mask = cv2.bitwise_or(road_mask, roadline_mask)

    mask_bgr = np.zeros((tag_image.shape[0], tag_image.shape[1], 3), dtype=np.uint8)
    mask_bgr[:, :, 0] = road_mask
    mask_bgr[:, :, 1] = roadline_mask
    mask_bgr[:, :, 2] = combined_mask

    return mask_bgr


# ============================================================
# MobileNet pitch model
# ============================================================

def build_mobilenet_regressor():
    """Create MobileNetV3-small with one regression output."""
    model = torchvision.models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 1)
    return model


def load_pitch_model(model_path, device):
    """Load trained pitch model checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Train first with 02_train_pitch_model.py."
        )

    checkpoint = torch.load(model_path, map_location=device)

    model = build_mobilenet_regressor()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    target_column = checkpoint.get("target_column", "unknown")
    image_size = int(checkpoint.get("image_size", DEFAULT_MODEL_IMAGE_SIZE))
    val_metrics = checkpoint.get("val_metrics", {})

    print(f"Loaded model: {os.path.abspath(model_path)}")
    print(f"Model target column: {target_column}")
    print(f"Model image size: {image_size}")

    if val_metrics:
        print(
            "Validation metrics: "
            f"MAE={val_metrics.get('mae', float('nan')):.4f} deg | "
            f"RMSE={val_metrics.get('rmse', float('nan')):.4f} deg"
        )

    return model, image_size, target_column


def make_preprocess(image_size):
    """Resize mask and convert it to tensor."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


@torch.no_grad()
def predict_pitch_from_mask(model, preprocess, mask_bgr, device):
    """Run MobileNet pitch prediction on one mask."""
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(mask_rgb).convert("RGB")

    tensor = preprocess(pil_image)
    tensor = tensor.unsqueeze(0).to(device)

    output = model(tensor)
    return float(output.detach().cpu().numpy().reshape(-1)[0])


# ============================================================
# GNSS pitch baseline
# ============================================================

def geodetic_horizontal_distance_m(lat1, lon1, lat2, lon2):
    """Approximate GNSS horizontal distance in meters."""
    earth_radius_m = 6371000.0

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad)
        * math.sin(delta_lon / 2.0) ** 2
    )

    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return earth_radius_m * c


def estimate_gnss_pitch_deg(previous_gnss, current_gnss):
    """Estimate pitch from altitude change over horizontal displacement."""
    if previous_gnss is None or current_gnss is None:
        return None

    horizontal_distance = geodetic_horizontal_distance_m(
        previous_gnss["latitude"],
        previous_gnss["longitude"],
        current_gnss["latitude"],
        current_gnss["longitude"],
    )

    # Avoid instability while stopped.
    if horizontal_distance < GNSS_PITCH_MIN_HORIZONTAL_DISTANCE:
        return None

    altitude_delta = current_gnss["altitude"] - previous_gnss["altitude"]
    pitch_deg = math.degrees(math.atan2(altitude_delta, horizontal_distance))

    # Drop unrealistic spikes.
    if abs(pitch_deg) > MAX_REASONABLE_GNSS_PITCH_DEG:
        return None

    return pitch_deg


# ============================================================
# Spawn selection
# ============================================================

def score_spawn_point(spawn_point, world_map, map_center):
    """Score spawn points by road usability and pitch variation."""
    waypoint = world_map.get_waypoint(
        spawn_point.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving
    )

    if waypoint is None or waypoint.is_junction:
        return -1e9, {}

    center_x, center_y = map_center
    distance_from_center = math.hypot(
        spawn_point.location.x - center_x,
        spawn_point.location.y - center_y
    )

    score = distance_from_center * 0.05

    details = {
        "distance_from_center": distance_from_center,
        "straightness_penalty": 0.0,
        "grade_variation": 0.0,
        "usable_ahead_count": 0,
    }

    base_yaw = waypoint.transform.rotation.yaw
    base_pitch = wrap_angle_180(waypoint.transform.rotation.pitch)
    ahead_pitches = []

    for distance in SPAWN_LOOKAHEAD_DISTANCES:
        next_waypoints = waypoint.next(distance)

        if not next_waypoints:
            score -= 25.0
            continue

        ahead = next_waypoints[0]

        if ahead.is_junction:
            score -= 15.0
            continue

        details["usable_ahead_count"] += 1
        score += 8.0

        yaw_delta = abs(wrap_angle_180(ahead.transform.rotation.yaw - base_yaw))
        details["straightness_penalty"] += yaw_delta
        score -= yaw_delta * 0.4

        ahead_pitch = wrap_angle_180(ahead.transform.rotation.pitch)
        ahead_pitches.append(ahead_pitch)

    if ahead_pitches:
        grade_variation = max(ahead_pitches) - min(ahead_pitches)
        details["grade_variation"] = grade_variation
        score += min(grade_variation, 6.0) * 3.0
        score += min(abs(base_pitch), 5.0)

    return score, details


def rank_test_spawn_points(spawn_points, world_map):
    """Rank CARLA spawn points for repeatable test route selection."""
    if not spawn_points:
        raise RuntimeError("No spawn points found in this map.")

    center_x = sum(sp.location.x for sp in spawn_points) / len(spawn_points)
    center_y = sum(sp.location.y for sp in spawn_points) / len(spawn_points)
    map_center = (center_x, center_y)

    scored_spawn_points = []

    for index, spawn_point in enumerate(spawn_points):
        score, details = score_spawn_point(spawn_point, world_map, map_center)
        scored_spawn_points.append((score, index, spawn_point, details))

    return sorted(scored_spawn_points, key=lambda item: item[0], reverse=True)


# ============================================================
# Kalman filter
# ============================================================

class SimplePitchKalmanFilter:
    """1-state KF: IMU gyro predicts pitch, camera ML corrects it."""

    def __init__(self):
        # Filter starts uninitialized.
        self.initialized = False

        # Current pitch estimate in degrees.
        self.theta_deg = 0.0

        # Current estimate uncertainty.
        self.P = 1.0

        # Process noise: IMU integration uncertainty.
        self.Q = 0.01

        # Measurement noise: camera pitch uncertainty.
        self.R = 0.01

    def step(self, dt, gyro_y_rad_s, camera_ml_pitch_deg, initial_pitch_deg=None):
        """Run one KF prediction/correction step."""

        # Need positive timestep.
        if dt is None or dt <= 0.0:
            return None, None

        # Need IMU gyro for prediction.
        if gyro_y_rad_s is None:
            return None, None

        # CARLA gyro is rad/s; pitch values are degrees.
        gyro_y_deg_s = math.degrees(gyro_y_rad_s)

        # Initialize once.
        if not self.initialized:
            # Ground truth is only used at startup.
            if initial_pitch_deg is not None:
                self.theta_deg = float(initial_pitch_deg)

            # Fallback if true pitch is unavailable.
            elif camera_ml_pitch_deg is not None:
                self.theta_deg = float(camera_ml_pitch_deg)

            # Last fallback.
            else:
                self.theta_deg = 0.0

            # Reset startup uncertainty.
            self.P = 1.0
            self.initialized = True

        # Predict pitch using gyro integration.
        theta_pred = self.theta_deg + gyro_y_deg_s * dt

        # Prediction increases uncertainty.
        P_pred = self.P + self.Q

        # Correct with camera if available.
        if camera_ml_pitch_deg is not None and math.isfinite(camera_ml_pitch_deg):
            # Camera minus prediction.
            innovation = camera_ml_pitch_deg - theta_pred

            # Kalman gain.
            K = P_pred / (P_pred + self.R)

            # Correct pitch.
            self.theta_deg = theta_pred + K * innovation

            # Correction reduces uncertainty.
            self.P = (1.0 - K) * P_pred

        else:
            # No camera, use prediction only.
            self.theta_deg = theta_pred
            self.P = P_pred

        # Return filtered pitch and used pitch rate.
        return float(self.theta_deg), float(gyro_y_deg_s)


# ============================================================
# Plotting
# ============================================================

def _plot_series(samples, specs, output_path, title):
    """Common plot helper."""
    if not samples:
        return False

    if plt is None:
        print("Matplotlib is not installed; skipping plot.")
        return False

    plt.figure(figsize=(12, 6))

    for key, label, style, linewidth, markersize in specs:
        xs, ys = finite_xy((sample["sim_time"], sample.get(key)) for sample in samples)
        if not xs:
            continue

        if style == ".":
            plt.plot(xs, ys, ".", label=label, markersize=markersize)
        else:
            plt.plot(xs, ys, style, label=label, linewidth=linewidth)

    plt.xlabel("Simulation time (s)")
    plt.ylabel("Pitch (deg)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return True


def save_camera_plot(samples, output_path):
    """Camera ML vs references."""
    specs = [
        ("true_pitch_deg", "True vehicle pitch", "-", 2.0, 0),
        ("waypoint_pitch_deg", "Waypoint road pitch", "-", 2.0, 0),
        ("camera_ml_pitch_deg", "Camera ML pitch", ".", 0, 2.5),
    ]
    return _plot_series(samples, specs, output_path, "Camera ML pitch vs CARLA references")


def save_gnss_plot(samples, output_path):
    """GNSS baseline vs references."""
    specs = [
        ("true_pitch_deg", "True vehicle pitch", "-", 2.0, 0),
        ("waypoint_pitch_deg", "Waypoint road pitch", "-", 2.0, 0),
        ("gnss_pitch_deg", "GNSS pitch", ".", 0, 2.5),
    ]
    return _plot_series(samples, specs, output_path, "GNSS pitch vs CARLA references")


def save_kf_plot(samples, output_path):
    """Final KF result vs references."""
    specs = [
        ("true_pitch_deg", "True vehicle pitch", "-", 2.0, 0),
        ("waypoint_pitch_deg", "Waypoint road pitch", "-", 2.0, 0),
        ("camera_ml_pitch_deg", "Camera ML measurement", ".", 0, 2.0),
        ("kf_pitch_deg", "Final KF pitch", "-", 2.0, 0),
    ]
    return _plot_series(samples, specs, output_path, "KF pitch estimate vs CARLA references")


def save_all_sensor_plot(samples, output_path):
    """All useful pitch estimates."""
    specs = [
        ("true_pitch_deg", "True vehicle pitch", "-", 2.0, 0),
        ("waypoint_pitch_deg", "Waypoint road pitch", "-", 2.0, 0),
        ("camera_ml_pitch_deg", "Camera ML pitch", ".", 0, 2.0),
        ("gnss_pitch_deg", "GNSS pitch", ".", 0, 2.0),
        ("kf_pitch_deg", "KF pitch", "-", 2.0, 0),
    ]
    return _plot_series(samples, specs, output_path, "All pitch estimates vs CARLA references")


def save_all_plots(samples, output_dir):
    """Save all report plots."""
    save_camera_plot(samples, os.path.join(output_dir, CAMERA_PLOT_NAME))
    save_gnss_plot(samples, os.path.join(output_dir, GNSS_PLOT_NAME))
    save_kf_plot(samples, os.path.join(output_dir, KF_PLOT_NAME))
    save_all_sensor_plot(samples, os.path.join(output_dir, ALL_SENSOR_PLOT_NAME))


# ============================================================
# Metrics
# ============================================================

def compute_metric_summary(samples, estimate_key, truth_key):
    """Compute simple error metrics."""
    pairs = []

    for sample in samples:
        estimate = sample.get(estimate_key)
        truth = sample.get(truth_key)

        if estimate is None or truth is None:
            continue

        if not math.isfinite(float(estimate)) or not math.isfinite(float(truth)):
            continue

        pairs.append((float(estimate), float(truth)))

    if not pairs:
        return None

    estimates = np.asarray([p[0] for p in pairs], dtype=np.float64)
    truths = np.asarray([p[1] for p in pairs], dtype=np.float64)

    errors = estimates - truths
    abs_errors = np.abs(errors)

    return {
        "count": int(len(pairs)),
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "max_abs": float(np.max(abs_errors)),
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
    }


def print_error_summary(samples):
    """Print report-useful metrics."""
    specs = [
        ("camera_ml_pitch_deg", "true_pitch_deg", "Camera ML vs true vehicle"),
        ("camera_ml_pitch_deg", "waypoint_pitch_deg", "Camera ML vs waypoint"),
        ("gnss_pitch_deg", "true_pitch_deg", "GNSS vs true vehicle"),
        ("gnss_pitch_deg", "waypoint_pitch_deg", "GNSS vs waypoint"),
        ("kf_pitch_deg", "true_pitch_deg", "KF vs true vehicle"),
        ("kf_pitch_deg", "waypoint_pitch_deg", "KF vs waypoint"),
    ]

    print("\nError summary")
    print("=============")

    any_summary = False

    for estimate_key, truth_key, label in specs:
        summary = compute_metric_summary(samples, estimate_key, truth_key)

        if summary is None:
            continue

        any_summary = True

        print(
            f"{label}: "
            f"N={summary['count']} | "
            f"MAE={summary['mae']:.4f} deg | "
            f"RMSE={summary['rmse']:.4f} deg | "
            f"max_abs={summary['max_abs']:.4f} deg | "
            f"mean_error={summary['mean_error']:.4f} deg"
        )

    if not any_summary:
        print("No valid estimates available for error summary.")


def save_error_summary_csv(samples, output_dir):
    """Save MAE/RMSE/max_abs table to CSV."""
    summary_path = os.path.join(output_dir, SUMMARY_CSV_NAME)

    specs = [
        ("camera_ml_pitch_deg", "true_pitch_deg", "Camera vs true"),
        ("camera_ml_pitch_deg", "waypoint_pitch_deg", "Camera vs waypoint"),
        ("gnss_pitch_deg", "true_pitch_deg", "GNSS vs true"),
        ("gnss_pitch_deg", "waypoint_pitch_deg", "GNSS vs waypoint"),
        ("kf_pitch_deg", "true_pitch_deg", "KF vs true"),
        ("kf_pitch_deg", "waypoint_pitch_deg", "KF vs waypoint"),
    ]

    with open(summary_path, mode="w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "estimate",
            "truth",
            "label",
            "valid_sample_count",
            "mae_deg",
            "rmse_deg",
            "max_abs_deg",
            "mean_error_deg",
            "std_error_deg",
        ])

        for estimate_key, truth_key, label in specs:
            summary = compute_metric_summary(samples, estimate_key, truth_key)

            if summary is None:
                continue

            writer.writerow([
                estimate_key,
                truth_key,
                label,
                summary["count"],
                summary["mae"],
                summary["rmse"],
                summary["max_abs"],
                summary["mean_error"],
                summary["std_error"],
            ])

    return summary_path

# ============================================================
# Args
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CARLA pitch estimation with ML camera and KF fusion."
    )

    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--duration", type=float, default=SIM_DURATION_SECONDS)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--save-masks", action="store_true")
    parser.add_argument("--full-csv", action="store_true", help="Log extra debug columns.")

    return parser.parse_args()


# ============================================================
# CSV helpers
# ============================================================

def get_csv_header(full_csv=False):
    """Return compact or full CSV header."""
    compact_header = [
        "sample_index",
        "sim_time",
        "dt",
        "speed_mps",
        "true_pitch_deg",
        "waypoint_pitch_deg",
        "camera_ml_pitch_deg",
        "gnss_pitch_deg",
        "gyro_y_deg_s",
        "kf_pitch_deg",
        "camera_error_vs_true_deg",
        "gnss_error_vs_true_deg",
        "kf_error_vs_true_deg",
    ]

    if not full_csv:
        return compact_header

    return compact_header + [
        "vehicle_x",
        "vehicle_y",
        "vehicle_z",
        "true_roll_deg",
        "true_yaw_deg",
        "camera_frame",
        "imu_frame",
        "gnss_frame",
        "road_pixel_count",
        "roadline_pixel_count",
        "combined_pixel_count",
        "accel_x",
        "accel_y",
        "accel_z",
        "gyro_x_rad_s",
        "gyro_y_rad_s",
        "gyro_z_rad_s",
        "compass",
        "gnss_latitude",
        "gnss_longitude",
        "gnss_altitude",
        "model_target_column",
        "model_image_size",
    ]


def build_csv_row(
    full_csv,
    sample_index,
    sim_time,
    dt,
    speed_mps,
    true_pitch_deg,
    waypoint_pitch_deg,
    camera_ml_pitch_deg,
    gnss_pitch_deg,
    gyro_y_deg_s,
    kf_pitch_deg,
    loc,
    true_roll_deg,
    true_yaw_deg,
    camera_frame,
    imu_frame,
    gnss_frame,
    road_pixel_count,
    roadline_pixel_count,
    combined_pixel_count,
    accel,
    gyro,
    compass,
    model_target_column,
    model_image_size,
):
    """Build compact or full CSV row."""
    camera_error = safe_error(camera_ml_pitch_deg, true_pitch_deg)
    gnss_error = safe_error(gnss_pitch_deg, true_pitch_deg)
    kf_error = safe_error(kf_pitch_deg, true_pitch_deg)

    row = [
        sample_index,
        sim_time,
        dt,
        speed_mps,
        true_pitch_deg,
        waypoint_pitch_deg,
        camera_ml_pitch_deg,
        gnss_pitch_deg,
        gyro_y_deg_s,
        kf_pitch_deg,
        camera_error,
        gnss_error,
        kf_error,
    ]

    if not full_csv:
        return row

    row += [
        loc.x,
        loc.y,
        loc.z,
        true_roll_deg,
        true_yaw_deg,
        camera_frame,
        imu_frame,
        gnss_frame,
        road_pixel_count,
        roadline_pixel_count,
        combined_pixel_count,
        accel.x if accel else None,
        accel.y if accel else None,
        accel.z if accel else None,
        gyro.x if gyro else None,
        gyro.y if gyro else None,
        gyro.z if gyro else None,
        compass,
        latest_gnss["latitude"],
        latest_gnss["longitude"],
        latest_gnss["altitude"],
        model_target_column,
        model_image_size,
    ]

    return row


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.save_masks:
        os.makedirs(os.path.join(args.output_dir, "masks"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nCARLA pitch estimation with ML camera and KF")
    print("===========================================")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    model, model_image_size, model_target_column = load_pitch_model(
        model_path=args.model_path,
        device=device,
    )

    preprocess = make_preprocess(model_image_size)

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)

    world = client.load_world(MAP_NAME)
    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager()

    actors = []
    samples = []

    previous_sim_time = None
    previous_gnss_measurement = None
    previous_gnss_frame = None
    last_processed_semantic_frame = None

    kf = SimplePitchKalmanFilter()

    try:
        print("\nConnected to CARLA")
        world_map = world.get_map()
        print(f"Map: {world_map.name}")

        # Synchronous mode keeps sensor timing repeatable.
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)

        blueprint_library = world.get_blueprint_library()

        # -------------------------
        # Vehicle
        # -------------------------
        vehicle_bp = random.choice(blueprint_library.filter("vehicle.tesla.model3"))
        spawn_points = world_map.get_spawn_points()

        if not spawn_points:
            raise RuntimeError("No spawn points found in this map.")

        vehicle = None

        if FORCE_BEST_TEST_SPAWN:
            ranked_spawn_points = rank_test_spawn_points(spawn_points, world_map)

            for spawn_score, spawn_index, spawn_point, spawn_details in ranked_spawn_points:
                vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
                if vehicle is not None:
                    break
        else:
            spawn_index = random.randrange(len(spawn_points))
            spawn_point = spawn_points[spawn_index]
            spawn_score = None
            spawn_details = {}
            vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

        if vehicle is None:
            raise RuntimeError("Could not spawn vehicle.")

        actors.append(vehicle)
        vehicle.set_autopilot(True, traffic_manager.get_port())

        print(f"Vehicle spawned: {vehicle.type_id}")
        print(
            f"Spawn: x={spawn_point.location.x:.2f}, "
            f"y={spawn_point.location.y:.2f}, z={spawn_point.location.z:.2f}"
        )
        print(f"Spawn index: {spawn_index}")

        if spawn_score is not None:
            print(
                f"Spawn score={spawn_score:.2f} | "
                f"grade variation={spawn_details['grade_variation']:.3f} deg"
            )

        # -------------------------
        # Semantic camera
        # -------------------------
        semantic_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
        semantic_bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
        semantic_bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
        semantic_bp.set_attribute("fov", str(CAMERA_FOV))
        semantic_bp.set_attribute("sensor_tick", str(CAMERA_SENSOR_TICK))

        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=2.4),
            carla.Rotation(pitch=0.0)
        )

        semantic_camera = world.spawn_actor(
            semantic_bp,
            camera_transform,
            attach_to=vehicle
        )
        actors.append(semantic_camera)
        semantic_camera.listen(semantic_callback)

        print("Semantic camera attached")

        # -------------------------
        # IMU
        # -------------------------
        imu_bp = blueprint_library.find("sensor.other.imu")
        imu_bp.set_attribute("sensor_tick", str(IMU_SENSOR_TICK))

        imu = world.spawn_actor(
            imu_bp,
            carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0)),
            attach_to=vehicle
        )
        actors.append(imu)
        imu.listen(imu_callback)

        print("IMU attached")

        # -------------------------
        # GNSS
        # -------------------------
        gnss_bp = blueprint_library.find("sensor.other.gnss")
        gnss_bp.set_attribute("sensor_tick", str(GNSS_SENSOR_TICK))

        gnss = world.spawn_actor(
            gnss_bp,
            carla.Transform(carla.Location(x=0.0, y=0.0, z=0.0)),
            attach_to=vehicle
        )
        actors.append(gnss)
        gnss.listen(gnss_callback)

        print("GNSS attached")

        # Sensor warm-up.
        for _ in range(10):
            world.tick()

        csv_path = os.path.join(args.output_dir, CSV_NAME)

        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(get_csv_header(args.full_csv))

            print("\nLogging started.")
            print(f"CSV: {os.path.abspath(csv_path)}")
            print("Press Ctrl+C to stop. Press q on the mask window to quit.\n")

            start_wall_time = time.time()
            sample_index = 0

            while True:
                world.tick()

                if args.duration is not None:
                    if time.time() - start_wall_time > args.duration:
                        break

                if latest_semantic["mask_bgr"] is None:
                    continue

                semantic_frame = latest_semantic["frame"]

                # Process each camera frame once.
                if semantic_frame == last_processed_semantic_frame:
                    continue

                last_processed_semantic_frame = semantic_frame

                sim_snapshot = world.get_snapshot()
                sim_time = sim_snapshot.timestamp.elapsed_seconds

                # Time step.
                dt = None
                if previous_sim_time is not None:
                    dt = sim_time - previous_sim_time
                previous_sim_time = sim_time

                # Vehicle truth/reference.
                transform = vehicle.get_transform()
                loc = transform.location
                rot = transform.rotation

                speed_mps = get_vehicle_speed_mps(vehicle)
                true_pitch_deg = wrap_angle_180(rot.pitch)
                true_roll_deg = wrap_angle_180(rot.roll)
                true_yaw_deg = wrap_angle_180(rot.yaw)

                # Road waypoint reference.
                current_waypoint = world_map.get_waypoint(
                    loc,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving
                )

                waypoint_pitch_deg = None
                if current_waypoint is not None:
                    waypoint_pitch_deg = wrap_angle_180(
                        current_waypoint.transform.rotation.pitch
                    )

                # Camera ML pitch.
                mask_bgr = latest_semantic["mask_bgr"].copy()

                camera_ml_pitch_deg = predict_pitch_from_mask(
                    model=model,
                    preprocess=preprocess,
                    mask_bgr=mask_bgr,
                    device=device,
                )

                road_pixel_count = int(np.count_nonzero(mask_bgr[:, :, 0] > 0))
                roadline_pixel_count = int(np.count_nonzero(mask_bgr[:, :, 1] > 0))
                combined_pixel_count = int(np.count_nonzero(mask_bgr[:, :, 2] > 0))

                # IMU pitch-rate input.
                accel = latest_imu["accel"]
                gyro = latest_imu["gyro"]
                compass = latest_imu["compass"]

                gyro_y_rad_s = gyro.y if gyro is not None else None
                gyro_y_deg_s = math.degrees(gyro_y_rad_s) if gyro_y_rad_s is not None else None

                # GNSS pitch baseline.
                gnss_pitch_deg = None

                if latest_gnss["latitude"] is not None:
                    current_gnss_measurement = {
                        "frame": latest_gnss["frame"],
                        "timestamp": latest_gnss["timestamp"],
                        "latitude": latest_gnss["latitude"],
                        "longitude": latest_gnss["longitude"],
                        "altitude": latest_gnss["altitude"],
                    }

                    if latest_gnss["frame"] != previous_gnss_frame:
                        gnss_pitch_deg = estimate_gnss_pitch_deg(
                            previous_gnss_measurement,
                            current_gnss_measurement,
                        )

                        # Keep last valid GNSS point as baseline anchor.
                        if previous_gnss_measurement is None or gnss_pitch_deg is not None:
                            previous_gnss_measurement = current_gnss_measurement

                        previous_gnss_frame = latest_gnss["frame"]

                # Kalman filter fusion.
                if gyro_y_rad_s is not None and dt is not None and dt > 0.0:
                    kf_pitch_deg, kf_pitch_rate_deg_s = kf.step(
                        dt=dt,
                        gyro_y_rad_s=gyro_y_rad_s,
                        camera_ml_pitch_deg=camera_ml_pitch_deg,
                        initial_pitch_deg=true_pitch_deg,
                    )
                else:
                    kf_pitch_deg = None
                    kf_pitch_rate_deg_s = None

                # Write CSV.
                writer.writerow(
                    build_csv_row(
                        full_csv=args.full_csv,
                        sample_index=sample_index,
                        sim_time=sim_time,
                        dt=dt,
                        speed_mps=speed_mps,
                        true_pitch_deg=true_pitch_deg,
                        waypoint_pitch_deg=waypoint_pitch_deg,
                        camera_ml_pitch_deg=camera_ml_pitch_deg,
                        gnss_pitch_deg=gnss_pitch_deg,
                        gyro_y_deg_s=gyro_y_deg_s,
                        kf_pitch_deg=kf_pitch_deg,
                        loc=loc,
                        true_roll_deg=true_roll_deg,
                        true_yaw_deg=true_yaw_deg,
                        camera_frame=latest_semantic["frame"],
                        imu_frame=latest_imu["frame"],
                        gnss_frame=latest_gnss["frame"],
                        road_pixel_count=road_pixel_count,
                        roadline_pixel_count=roadline_pixel_count,
                        combined_pixel_count=combined_pixel_count,
                        accel=accel,
                        gyro=gyro,
                        compass=compass,
                        model_target_column=model_target_column,
                        model_image_size=model_image_size,
                    )
                )
                f.flush()

                # Store plot/metric sample.
                samples.append({
                    "sim_time": sim_time,
                    "true_pitch_deg": true_pitch_deg,
                    "waypoint_pitch_deg": waypoint_pitch_deg,
                    "camera_ml_pitch_deg": camera_ml_pitch_deg,
                    "gnss_pitch_deg": gnss_pitch_deg,
                    "kf_pitch_deg": kf_pitch_deg,
                })

                # Update plots during run.
                if (
                    SAVE_PLOTS_DURING_RUN
                    and len(samples) >= 2
                    and len(samples) % PLOT_SAVE_EVERY_N_FRAMES == 0
                ):
                    save_all_plots(samples, args.output_dir)

                # Terminal status.
                gnss_text = (
                    f"gnss={gnss_pitch_deg:7.3f}"
                    if gnss_pitch_deg is not None
                    else "gnss=   None"
                )

                kf_text = (
                    f"kf={kf_pitch_deg:7.3f}"
                    if kf_pitch_deg is not None
                    else "kf=   None"
                )

                print(
                    f"\r"
                    f"samples={sample_index:06d} | "
                    f"t={sim_time:7.2f}s | "
                    f"GT={true_pitch_deg:7.3f} | "
                    f"wp={waypoint_pitch_deg if waypoint_pitch_deg is not None else 0.0:7.3f} | "
                    f"cam={camera_ml_pitch_deg:7.3f} | "
                    f"{gnss_text} | "
                    f"gyro_y={gyro_y_deg_s if gyro_y_deg_s is not None else 0.0:8.3f} | "
                    f"{kf_text}",
                    end=""
                )

                # Optional OpenCV display.
                if DISPLAY_MASK and not args.no_display:
                    debug_view = mask_bgr.copy()

                    cv2.putText(
                        debug_view,
                        f"GT: {true_pitch_deg:.3f} deg",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    if waypoint_pitch_deg is not None:
                        cv2.putText(
                            debug_view,
                            f"Waypoint: {waypoint_pitch_deg:.3f} deg",
                            (30, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                    cv2.putText(
                        debug_view,
                        f"Camera ML: {camera_ml_pitch_deg:.3f} deg",
                        (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    if kf_pitch_deg is not None:
                        cv2.putText(
                            debug_view,
                            f"KF: {kf_pitch_deg:.3f} deg",
                            (30, 160),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                    cv2.imshow("CARLA Pitch - ML Camera Mask", debug_view)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # Optional mask saving.
                if args.save_masks and sample_index % MASK_SAVE_EVERY_N_FRAMES == 0:
                    mask_path = os.path.join(
                        args.output_dir,
                        "masks",
                        f"mask_{sample_index:06d}.png",
                    )
                    cv2.imwrite(mask_path, mask_bgr)

                sample_index += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        print("\nSaving outputs...")

        save_all_plots(samples, args.output_dir)
        print_error_summary(samples)
        summary_csv_path = save_error_summary_csv(samples, args.output_dir)

        print("\nCleaning up CARLA...")

        for actor in actors:
            if actor is not None:
                try:
                    actor.destroy()
                except Exception as exc:
                    print(f"Actor cleanup warning: {exc}")

        try:
            world.apply_settings(original_settings)
        except Exception as exc:
            print(f"World settings cleanup warning: {exc}")

        try:
            traffic_manager.set_synchronous_mode(False)
        except Exception as exc:
            print(f"Traffic manager cleanup warning: {exc}")

        cv2.destroyAllWindows()

        print(f"\nCSV: {os.path.abspath(os.path.join(args.output_dir, CSV_NAME))}")
        print(f"Error summary CSV: {os.path.abspath(summary_csv_path)}")
        print(f"Camera plot: {os.path.abspath(os.path.join(args.output_dir, CAMERA_PLOT_NAME))}")
        print(f"GNSS plot: {os.path.abspath(os.path.join(args.output_dir, GNSS_PLOT_NAME))}")
        print(f"KF plot: {os.path.abspath(os.path.join(args.output_dir, KF_PLOT_NAME))}")
        print(f"All sensor plot: {os.path.abspath(os.path.join(args.output_dir, ALL_SENSOR_PLOT_NAME))}")
        print(f"Run folder: {os.path.abspath(args.output_dir)}")
        print("Done.")


if __name__ == "__main__":
    main()
