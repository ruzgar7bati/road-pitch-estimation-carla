import carla
import random
import time
import csv
import os
import math

import numpy as np
import cv2


# =========================
# Settings
# =========================

OUTPUT_DIR_NAME = "dataset_pitch"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, OUTPUT_DIR_NAME)

TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
VAL_DIR = os.path.join(OUTPUT_DIR, "val")
TRAIN_MASK_DIR = os.path.join(TRAIN_DIR, "masks")
VAL_MASK_DIR = os.path.join(VAL_DIR, "masks")

TRAIN_CSV_NAME = "train.csv"
VAL_CSV_NAME = "val.csv"
ALL_CSV_NAME = "dataset_all.csv"

MAP_NAME = "Town05"

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FOV = 90
SENSOR_TICK = 0.05  # 20 Hz

# Save every N semantic camera frames.
# 1 = save every frame, 2 = save every other frame, etc.
SAVE_EVERY_N_FRAMES = 1

# Set None for manual stop with Ctrl+C or q.
SIM_DURATION_SECONDS = 300

DISPLAY_MASK = True

# Split settings.
TRAIN_RATIO = 0.80
RANDOM_SPLIT_SEED = 42

# Important:
# True random frame split can leak nearly identical adjacent frames into both train and val.
# For a "quick pass-the-class" demo, random split is okay.
# For a cleaner validation, set this to "block".
#
# "random": each frame independently goes to train/val with 80/20 probability.
# "block": saves blocks of frames; 4 train blocks, then 1 val block, repeating.
SPLIT_MODE = "random"
BLOCK_SIZE = 100

# Same idea as your original script: use the best-looking pitch/ramp spawn.
FORCE_BEST_TEST_SPAWN = True
SPAWN_LOOKAHEAD_DISTANCES = [10.0, 20.0, 30.0, 40.0, 60.0]

# CARLA 0.9.16 semantic segmentation tag values.
# Important: CARLA changed semantic tags from 0.9.13 to 0.9.14.
# In CARLA 0.9.16:
#   Roads    = 1
#   RoadLine = 24
# In the semantic camera raw data, the tag is stored in the red channel.
SEMANTIC_TAG_ROAD = 1
SEMANTIC_TAG_ROAD_LINE = 24

# Make lane/road lines thicker so MobileNet can see them more easily.
DILATE_ROAD_LINES = True
ROAD_LINE_DILATE_KERNEL_SIZE = 3
ROAD_LINE_DILATE_ITERATIONS = 1


# =========================
# Shared sensor storage
# =========================

latest_semantic = {
    "frame": None,
    "timestamp": None,
    "tag_image": None,
    "mask_bgr": None,
}


# =========================
# Utility functions
# =========================

def wrap_angle_180(angle_deg):
    """
    Wrap an angle to [-180, 180].
    """
    if angle_deg is None:
        return None

    return ((angle_deg + 180.0) % 360.0) - 180.0


def get_vehicle_speed_mps(vehicle):
    """
    Return vehicle speed in meters/second.
    """
    velocity = vehicle.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def choose_split(sample_index, rng):
    """
    Choose whether this sample goes to train or val.

    Returns:
        split_name, split_mask_dir, relative_mask_path_prefix
    """
    if SPLIT_MODE == "random":
        split_name = "train" if rng.random() < TRAIN_RATIO else "val"

    elif SPLIT_MODE == "block":
        # 80/20 by repeating 5 blocks:
        # block positions 0,1,2,3 -> train
        # block position 4 -> val
        block_index = sample_index // BLOCK_SIZE
        block_position = block_index % 5
        split_name = "val" if block_position == 4 else "train"

    else:
        raise ValueError(f"Unknown SPLIT_MODE: {SPLIT_MODE}")

    if split_name == "train":
        return split_name, TRAIN_MASK_DIR, "train/masks"

    return split_name, VAL_MASK_DIR, "val/masks"


def make_road_roadline_mask_from_tags(tag_image):
    """
    Convert CARLA semantic tag image into a 3-channel BGR mask.

    Channel meaning:
        B channel: road mask
        G channel: road-line mask
        R channel: combined road + road-line mask

    Saved PNGs therefore contain only road and road-line information.
    Background is black.
    """
    road_mask = (tag_image == SEMANTIC_TAG_ROAD).astype(np.uint8) * 255
    roadline_mask = (tag_image == SEMANTIC_TAG_ROAD_LINE).astype(np.uint8) * 255

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


def semantic_callback(image):
    """
    CARLA semantic segmentation image callback.

    CARLA raw_data is BGRA.
    For semantic segmentation, the class tag is encoded in the red channel.
    """
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))

    tag_image = array[:, :, 2]  # Red channel in BGRA
    mask_bgr = make_road_roadline_mask_from_tags(tag_image)

    latest_semantic["frame"] = image.frame
    latest_semantic["timestamp"] = image.timestamp
    latest_semantic["tag_image"] = tag_image
    latest_semantic["mask_bgr"] = mask_bgr


# =========================
# Spawn-point scoring
# Copied/adapted from your original script.
# =========================

def score_spawn_point(spawn_point, world_map, map_center):
    """
    Score spawn points for repeatable pitch experiments.
    """
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


def get_csv_header():
    return [
        "sample_index",
        "split",
        "semantic_frame",
        "semantic_time",
        "sim_time",
        "mask_path",
        "vehicle_x",
        "vehicle_y",
        "vehicle_z",
        "true_vehicle_pitch_deg",
        "waypoint_pitch_deg",
        "true_roll_deg",
        "true_yaw_deg",
        "speed_mps",
        "town",
        "spawn_index",
        "camera_width",
        "camera_height",
        "camera_fov",
        "road_pixel_count",
        "roadline_pixel_count",
        "combined_pixel_count",
    ]


# =========================
# Main
# =========================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TRAIN_MASK_DIR, exist_ok=True)
    os.makedirs(VAL_MASK_DIR, exist_ok=True)

    rng = random.Random(RANDOM_SPLIT_SEED)

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(10.0)

    world = client.load_world(MAP_NAME)
    original_settings = world.get_settings()
    traffic_manager = client.get_trafficmanager()

    actors = []
    last_saved_semantic_frame = None
    saved_count = 0
    train_count = 0
    val_count = 0

    pitch_min_vehicle = None
    pitch_max_vehicle = None
    pitch_min_waypoint = None
    pitch_max_waypoint = None

    try:
        print("\nConnected to CARLA")
        world_map = world.get_map()
        print(f"Map: {world_map.name}")

        # Synchronous mode keeps image/label logging aligned.
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        world.apply_settings(settings)
        traffic_manager.set_synchronous_mode(True)

        blueprint_library = world.get_blueprint_library()

        # =========================
        # Spawn vehicle
        # =========================

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
            raise RuntimeError("Could not spawn the vehicle at any selected spawn point.")

        actors.append(vehicle)
        vehicle.set_autopilot(True, traffic_manager.get_port())

        print(f"Vehicle spawned: {vehicle.type_id}")
        print(
            f"Spawn location: x={spawn_point.location.x:.2f}, "
            f"y={spawn_point.location.y:.2f}, z={spawn_point.location.z:.2f}"
        )
        print(f"Spawn index: {spawn_index}")

        if spawn_score is not None:
            print(
                f"Forced test spawn score: {spawn_score:.2f} | "
                f"center_dist={spawn_details['distance_from_center']:.2f} m | "
                f"usable_ahead={spawn_details['usable_ahead_count']} | "
                f"grade_variation={spawn_details['grade_variation']:.3f} deg"
            )

        # =========================
        # Attach semantic camera
        # =========================

        semantic_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
        semantic_bp.set_attribute("image_size_x", str(CAMERA_WIDTH))
        semantic_bp.set_attribute("image_size_y", str(CAMERA_HEIGHT))
        semantic_bp.set_attribute("fov", str(CAMERA_FOV))
        semantic_bp.set_attribute("sensor_tick", str(SENSOR_TICK))

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

        print("Semantic segmentation camera attached")
        print(f"Camera resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
        print(f"Camera horizontal FOV: {CAMERA_FOV} deg")
        print("Saving only road + road-line mask images")
        print(f"Semantic tags: road={SEMANTIC_TAG_ROAD}, roadline={SEMANTIC_TAG_ROAD_LINE}")
        print(f"Split mode: {SPLIT_MODE}, train ratio target: {TRAIN_RATIO:.2f}")

        # Warm-up ticks so the sensor starts producing frames.
        for _ in range(10):
            world.tick()

        # =========================
        # CSV logging
        # =========================

        train_csv_path = os.path.join(OUTPUT_DIR, TRAIN_CSV_NAME)
        val_csv_path = os.path.join(OUTPUT_DIR, VAL_CSV_NAME)
        all_csv_path = os.path.join(OUTPUT_DIR, ALL_CSV_NAME)

        with (
            open(train_csv_path, mode="w", newline="") as train_f,
            open(val_csv_path, mode="w", newline="") as val_f,
            open(all_csv_path, mode="w", newline="") as all_f,
        ):
            train_writer = csv.writer(train_f)
            val_writer = csv.writer(val_f)
            all_writer = csv.writer(all_f)

            header = get_csv_header()
            train_writer.writerow(header)
            val_writer.writerow(header)
            all_writer.writerow(header)

            print("\nDataset collection started.")
            print(f"Output folder: {os.path.abspath(OUTPUT_DIR)}")
            print(f"Train CSV: {os.path.abspath(train_csv_path)}")
            print(f"Val CSV: {os.path.abspath(val_csv_path)}")
            print(f"All CSV: {os.path.abspath(all_csv_path)}")
            print("Press Ctrl+C to stop. Press q on the mask window to quit.\n")

            start_wall_time = time.time()

            while True:
                world.tick()

                if SIM_DURATION_SECONDS is not None:
                    if time.time() - start_wall_time > SIM_DURATION_SECONDS:
                        break

                if latest_semantic["mask_bgr"] is None:
                    continue

                semantic_frame = latest_semantic["frame"]

                # Save each semantic frame once.
                if semantic_frame == last_saved_semantic_frame:
                    continue

                last_saved_semantic_frame = semantic_frame

                # Optional downsampling by frame number.
                if semantic_frame % SAVE_EVERY_N_FRAMES != 0:
                    continue

                split_name, split_mask_dir, split_rel_prefix = choose_split(saved_count, rng)

                sim_snapshot = world.get_snapshot()
                sim_time = sim_snapshot.timestamp.elapsed_seconds

                transform = vehicle.get_transform()
                loc = transform.location
                rot = transform.rotation

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

                true_vehicle_pitch_deg = wrap_angle_180(rot.pitch)
                true_roll_deg = wrap_angle_180(rot.roll)
                true_yaw_deg = wrap_angle_180(rot.yaw)
                speed_mps = get_vehicle_speed_mps(vehicle)

                mask_bgr = latest_semantic["mask_bgr"].copy()

                road_mask = mask_bgr[:, :, 0] > 0
                roadline_mask = mask_bgr[:, :, 1] > 0
                combined_mask = mask_bgr[:, :, 2] > 0

                road_pixel_count = int(np.count_nonzero(road_mask))
                roadline_pixel_count = int(np.count_nonzero(roadline_mask))
                combined_pixel_count = int(np.count_nonzero(combined_mask))

                mask_filename = f"mask_{saved_count:06d}.png"
                mask_path_abs = os.path.join(split_mask_dir, mask_filename)
                mask_path_rel = f"{split_rel_prefix}/{mask_filename}"

                cv2.imwrite(mask_path_abs, mask_bgr)

                row = [
                    saved_count,
                    split_name,
                    semantic_frame,
                    latest_semantic["timestamp"],
                    sim_time,
                    mask_path_rel,
                    loc.x,
                    loc.y,
                    loc.z,
                    true_vehicle_pitch_deg,
                    waypoint_pitch_deg,
                    true_roll_deg,
                    true_yaw_deg,
                    speed_mps,
                    MAP_NAME,
                    spawn_index,
                    CAMERA_WIDTH,
                    CAMERA_HEIGHT,
                    CAMERA_FOV,
                    road_pixel_count,
                    roadline_pixel_count,
                    combined_pixel_count,
                ]

                all_writer.writerow(row)

                if split_name == "train":
                    train_writer.writerow(row)
                    train_count += 1
                else:
                    val_writer.writerow(row)
                    val_count += 1

                train_f.flush()
                val_f.flush()
                all_f.flush()

                if pitch_min_vehicle is None or true_vehicle_pitch_deg < pitch_min_vehicle:
                    pitch_min_vehicle = true_vehicle_pitch_deg
                if pitch_max_vehicle is None or true_vehicle_pitch_deg > pitch_max_vehicle:
                    pitch_max_vehicle = true_vehicle_pitch_deg

                if waypoint_pitch_deg is not None:
                    if pitch_min_waypoint is None or waypoint_pitch_deg < pitch_min_waypoint:
                        pitch_min_waypoint = waypoint_pitch_deg
                    if pitch_max_waypoint is None or waypoint_pitch_deg > pitch_max_waypoint:
                        pitch_max_waypoint = waypoint_pitch_deg

                total_count = train_count + val_count
                current_train_ratio = train_count / total_count if total_count > 0 else 0.0

                print(
                    f"\r"
                    f"samples={saved_count:06d} | "
                    f"train={train_count:06d} val={val_count:06d} "
                    f"train_ratio={current_train_ratio:5.3f} | "
                    f"t={sim_time:7.2f}s | "
                    f"vehicle_pitch={true_vehicle_pitch_deg:7.3f} deg | "
                    f"waypoint_pitch={waypoint_pitch_deg if waypoint_pitch_deg is not None else 0.0:7.3f} deg | "
                    f"speed={speed_mps:5.2f} m/s | "
                    f"road_px={road_pixel_count:7d} | "
                    f"line_px={roadline_pixel_count:6d}",
                    end=""
                )

                if DISPLAY_MASK:
                    debug_view = mask_bgr.copy()

                    cv2.putText(
                        debug_view,
                        f"Split: {split_name}",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                    cv2.putText(
                        debug_view,
                        f"Vehicle pitch: {true_vehicle_pitch_deg:.3f} deg",
                        (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

                    if waypoint_pitch_deg is not None:
                        cv2.putText(
                            debug_view,
                            f"Waypoint pitch: {waypoint_pitch_deg:.3f} deg",
                            (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA
                        )

                    cv2.imshow("CARLA road + road-line mask", debug_view)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                saved_count += 1

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        print("\nCleaning up...")

        for actor in actors:
            if actor is not None:
                actor.destroy()

        world.apply_settings(original_settings)
        traffic_manager.set_synchronous_mode(False)
        cv2.destroyAllWindows()

        total_count = train_count + val_count
        train_ratio_actual = train_count / total_count if total_count > 0 else 0.0

        print(f"\nSaved samples: {saved_count}")
        print(f"Train samples: {train_count}")
        print(f"Val samples: {val_count}")
        print(f"Actual train ratio: {train_ratio_actual:.3f}")
        print(f"Dataset folder: {os.path.abspath(OUTPUT_DIR)}")

        if pitch_min_vehicle is not None and pitch_max_vehicle is not None:
            print(
                f"Vehicle pitch range: "
                f"{pitch_min_vehicle:.3f} to {pitch_max_vehicle:.3f} deg"
            )

        if pitch_min_waypoint is not None and pitch_max_waypoint is not None:
            print(
                f"Waypoint pitch range: "
                f"{pitch_min_waypoint:.3f} to {pitch_max_waypoint:.3f} deg"
            )

        print("Done.")


if __name__ == "__main__":
    main()
