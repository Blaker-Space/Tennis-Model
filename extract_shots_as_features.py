"""
Capture shots from annotation as a succession of features into CSV files.
We store only useful joints (no eyes/ears), as (y,x) pairs per frame.
"""

from argparse import ArgumentParser
from pathlib import Path
import os
import numpy as np
import pandas as pd
import cv2

from extract_human_pose import HumanPoseExtractor

# 13 joints * 2 (y,x) = 26 columns
COLUMNS = [
    "nose_y","nose_x",
    "left_shoulder_y","left_shoulder_x",
    "right_shoulder_y","right_shoulder_x",
    "left_elbow_y","left_elbow_x",
    "right_elbow_y","right_elbow_x",
    "left_wrist_y","left_wrist_x",
    "right_wrist_y","right_wrist_x",
    "left_hip_y","left_hip_x",
    "right_hip_y","right_hip_x",
    "left_knee_y","left_knee_x",
    "right_knee_y","right_knee_x",
    "left_ankle_y","left_ankle_x",
    "right_ankle_y","right_ankle_x",
]

DROP_KEYPOINTS = ["left_eye", "right_eye", "left_ear", "right_ear"]

# --- Tunables ---
NB_IMAGES = 30            # frames per snippet (fixed length per window)
HALF = NB_IMAGES // 2     # 15
CONF_THRESHOLD_MEAN = 0.20   # per-frame mean kp confidence to accept a frame
MIN_KEEP_FRAMES = 20         # save a window only if we collected at least this many frames
ZERO_BASED_ANNOT = False     # set True if your annotation FrameId starts at 0 while loop is 1-based

# indices for the 13 useful joints in MoveNet (17 total)
USE_IDXS = [0,5,6,7,8,9,10,11,12,13,14,15,16]  # nose, shoulders, elbows, wrists, hips, knees, ankles

def draw_shot(frame, shot):
    cv2.putText(
        frame, str(shot),
        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8, color=(0,165,255), thickness=2
    )

def normalize_annotations(df: pd.DataFrame) -> pd.DataFrame:
    """Accept either column order ('FrameId,Shot' or 'Shot,FrameId'), standardize names/order."""
    cols = [c.strip().lower() for c in df.columns]
    if "frameid" in cols and "shot" in cols:
        # keep, but normalize capitalization and order
        df = df.rename(columns={df.columns[cols.index("frameid")]: "FrameId",
                                df.columns[cols.index("shot")]: "Shot"})[["FrameId","Shot"]]
    elif cols == ["shot","frameid"]:
        df.columns = ["Shot","FrameId"]
        df = df[["FrameId","Shot"]]
    else:
        raise ValueError("Annotation CSV must have columns 'FrameId' and 'Shot' (any order).")

    # convert types
    df["FrameId"] = df["FrameId"].astype(int)
    df["Shot"] = df["Shot"].astype(str)
    if ZERO_BASED_ANNOT:
        df["FrameId"] = df["FrameId"] + 1  # align with FRAME_ID starting at 1
    return df.sort_values("FrameId").reset_index(drop=True)

def collapse_runs_to_centers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge consecutive frames with the same Shot into one row whose FrameId is the run center.
    Prevents heavy overlap when annotations mark every frame in a sequence.
    """
    if df.empty:
        return df
    rows = []
    start = 0
    n = len(df)
    while start < n:
        shot = df.iloc[start]["Shot"]
        end = start
        # strict consecutiveness of frames for the same label
        while end + 1 < n and df.iloc[end + 1]["Shot"] == shot and \
              df.iloc[end + 1]["FrameId"] == df.iloc[end]["FrameId"] + 1:
            end += 1
        center_idx = (start + end) // 2
        rows.append({"FrameId": int(df.iloc[center_idx]["FrameId"]), "Shot": shot})
        start = end + 1
    return pd.DataFrame(rows)

def resample_to_len(frames_list, target_len):
    """
    frames_list: list of (1, 26) arrays
    Return list of exactly target_len (1,26) arrays using linear index interpolation.
    """
    if len(frames_list) == target_len:
        return frames_list
    if len(frames_list) == 0:
        return []
    arr = np.concatenate(frames_list, axis=0)  # (n, 26)
    n = arr.shape[0]
    xs = np.linspace(0, n - 1, num=target_len)
    x0 = np.floor(xs).astype(int)
    x1 = np.clip(x0 + 1, 0, n - 1)
    w = xs - x0
    out = (1 - w)[:, None] * arr[x0] + w[:, None] * arr[x1]
    return [out[i:i+1, :] for i in range(target_len)]

if __name__ == "__main__":
    parser = ArgumentParser(description="Annotate (associate human pose to a tennis/pickleball shot)")
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("annotation", help="CSV with columns: FrameId,Shot OR Shot,FrameId")
    parser.add_argument("out", help="Output folder for per-shot CSV files")
    parser.add_argument("--show", action="store_const", const=True, default=False, help="Show frame UI")
    args = parser.parse_args()

    # Load annotations and normalize
    raw = pd.read_csv(args.annotation)
    shots = normalize_annotations(raw)
    shots = collapse_runs_to_centers(shots)

    # Prepare output directory
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"Could not open video: {args.video}"

    ret, frame = cap.read()
    assert ret, "Could not read first frame."

    # Pose extractor (uses sticky RoI from your modified file)
    human_pose_extractor = HumanPoseExtractor(frame.shape)

    # Optional UI pacing
    if args.show:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        if fps <= 1 or fps > 120:
            fps = 30
        delay_ms = max(1, int(1000 / fps))
    else:
        delay_ms = 1  # process as fast as possible

    # State
    FRAME_ID = 1
    CURRENT_ROW = 0
    shots_features = []  # buffer for the current window

    # Per-class running indices
    idx_map = {"forehand":1, "backhand":1, "serve":1, "neutral":1}

    def write_window(features_list, shot_label):
        """Write the accumulated NB_IMAGES x 26 matrix to CSV."""
        arr = np.concatenate(features_list, axis=0)  # shape (NB_IMAGES, 26)
        shots_df = pd.DataFrame(arr, columns=COLUMNS)
        shots_df["shot"] = np.full(NB_IMAGES, shot_label)

        label = str(shot_label).lower()
        if label not in idx_map:
            idx_map[label] = 1

        outpath = out_dir / f"{label}_{idx_map[label]:03d}.csv"
        idx_map[label] += 1

        shots_df.to_csv(outpath, index=False)
        print(f"Saved {label} window -> {outpath}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Pose for this frame
        human_pose_extractor.extract(frame)
        human_pose_extractor.discard(DROP_KEYPOINTS)
        human_pose_extractor.roi.update(human_pose_extractor.keypoints_pixels_frame)

        kps = human_pose_extractor.keypoints_with_scores.reshape(17, 3)
        feat_13x3 = kps[USE_IDXS, :]  # (13, 3)

        # Done with all labels?
        if CURRENT_ROW >= len(shots):
            # (Neutral windows after last label omitted for simplicity)
            if args.show:
                human_pose_extractor.draw_results_frame(frame)
                cv2.imshow("Frame", frame)
                if cv2.waitKey(delay_ms) & 0xFF == 27:
                    break
            else:
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            FRAME_ID += 1
            continue

        shot_frame = int(shots.iloc[CURRENT_ROW]["FrameId"])
        shot_label = shots.iloc[CURRENT_ROW]["Shot"]

        # Half-open bounds: exactly NB_IMAGES frames -> [start, end_excl)
        start = shot_frame - HALF
        end_excl = shot_frame + HALF

        # If we just reached the start of this window, clear buffer
        if FRAME_ID == start:
            shots_features = []

        # Inside the window?
        if start <= FRAME_ID < end_excl:
            # Only append frame if confidence is good; otherwise skip this frame
            if np.mean(feat_13x3[:, 2]) >= CONF_THRESHOLD_MEAN:
                feat_13x2 = feat_13x3[:, 0:2].reshape(1, 13 * 2)
                shots_features.append(feat_13x2)
                draw_shot(frame, shot_label)

            # If this is the last frame of the window, write or skip
            if FRAME_ID == end_excl - 1:
                if len(shots_features) >= MIN_KEEP_FRAMES:
                    shots_features = resample_to_len(shots_features, NB_IMAGES)
                    write_window(shots_features, shot_label)
                else:
                    print(f"Warning: window for '{shot_label}' has {len(shots_features)} frames (<{MIN_KEEP_FRAMES}); skipping.")
                shots_features = []
                CURRENT_ROW += 1

        # Show overlays if requested
        if args.show:
            human_pose_extractor.draw_results_frame(frame)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(delay_ms) & 0xFF == 27:
                break
        else:
            if cv2.waitKey(1) & 0xFF == 27:
                break

        FRAME_ID += 1

    cap.release()
    cv2.destroyAllWindows()