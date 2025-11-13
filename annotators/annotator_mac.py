import cv2
import csv
import os
import sys

# --- SHOT LABELS ---
SHOT_LABELS = {
    ord('w'): "SERVE",
    ord('d'): "FOREHAND",
    ord('a'): "BACKHAND",
}

def annotate_video(video_path):
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error opening video stream or file: {video_path}")
        sys.exit(1)

    # --- Create output filename based on input video name ---
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_csv = f"annotations_{base_name}.csv"

    csv_file = open(output_csv, mode='w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["frame_id", "shot_type"])

    print("🎾 Annotation started!")
    print("Controls:")
    print("   D : FOREHAND")
    print("   A : BACKHAND")
    print("   W : SERVE")
    print("   ESC : Exit and save")
    print("💡 You can hold down keys to annotate multiple frames.")

    current_label = None
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ End of video reached.")
            break

        display_frame = frame.copy()

        # Overlay text for feedback
        if current_label:
            cv2.putText(display_frame, f"{current_label}", (40, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        cv2.imshow("Annotator", display_frame)

        key = cv2.waitKey(10) & 0xFF

        # Update shot label based on pressed key
        if key in SHOT_LABELS:
            current_label = SHOT_LABELS[key]
            print(f"Frame {frame_id}: {current_label}")

        # If a label is active, annotate current frame
        if current_label:
            writer.writerow([frame_id, current_label])

        # ESC to exit
        if key == 27:
            print(f"💾 ESC pressed — saving {output_csv}...")
            break

        frame_id += 1

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"✅ Annotations saved to {output_csv}")

# --- MAIN ENTRY ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python annotator_mac.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    annotate_video(video_path)
