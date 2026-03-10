import bz2
import re
import tarfile
from io import BytesIO
from pathlib import Path

import pandas as pd
import yaml

from .tar_writer import WebDatasetWriter


def read_bz2_file(file_path: Path):
    """
    Reads a .bz2 compressed file and returns its decompressed content as bytes.
    """
    try:
        with bz2.open(file_path, "rb") as bz2f:
            return bz2f.read()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred while reading '{file_path}': {e}")
    return None


"""
def read_gaze_data(file_path):
    # Now read the whole file as raw text to get the gaze part
    gaze_data = []
    with open(file_path, "r") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")
            try:
                gaze_floats = list(map(float, parts[6:]))
            except ValueError:
                gaze_floats = [-1, -1]
            gaze_points = [
                [gaze_floats[i], gaze_floats[i + 1]]
                for i in range(0, len(gaze_floats), 2)
            ]
            gaze_data.append(gaze_points)
    # Add gaze column to dataframe
    return gaze_data

"""


def read_gaze_data(file_path):
    gaze_data = []
    actions = []

    with open(file_path, "r") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split(",")

            # Column 5 is action
            actions.append(int(parts[5]))

            try:
                gaze_floats = list(map(float, parts[6:]))
            except ValueError:
                gaze_floats = [-1, -1]

            gaze_points = [
                [gaze_floats[i], gaze_floats[i + 1]]
                for i in range(0, len(gaze_floats), 2)
            ]
            gaze_data.append(gaze_points)

    return gaze_data, actions


def extract_frame_number(member):
    """
    Extract the trailing number from filenames like 'JAW_3117023_17135.png'.
    Returns it as an int for proper numerical sorting.
    """
    match = re.search(r"_(\d+)\.png$", member.name)
    return int(match.group(1)) if match else 0


def extract_images_and_write_to_webdataset(
    tar_bz2_file, writer, eye_gaze, actions
) -> None:
    """
    Decompresses the .tar.bz2 file, extracts PNG images, and writes them to a WebDataset tar file.
    Sorts files based on the numeric part of the filename (e.g., JAW_3117023_17135.png → 17135).
    """
    decompressed_data = read_bz2_file(tar_bz2_file)
    if decompressed_data is None:
        return
    config_path = "configs/config.yaml"
    config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)
    tar_bytes = BytesIO(decompressed_data)

    with tarfile.open(fileobj=tar_bytes, mode="r:") as tar:
        members = sorted(
            [
                m
                for m in tar.getmembers()
                if m.isfile() and m.name.lower().endswith(".png")
            ],
            key=extract_frame_number,
        )
        # NOTE: The first member of tar (num=0) file is the info. So the images start from num=1

        for idx, member in enumerate(members, start=1):
            file_data = tar.extractfile(member).read()
            try:
                sample = {
                    "__key__": str(idx - 1),
                    "jpg": file_data,
                    "json": eye_gaze[idx - 1],
                    "action.cls": actions[idx],
                }
                writer.write(sample)
            except (ValueError, IndexError):
                print(f"Incorrect data format or mismatch for {member.name}")


def get_game_meta_data(game: str, config: dict) -> pd.DataFrame:
    """
    Reads meta data CSV from config and returns DataFrame grouped by subject_id with list of trial_ids for the given game.
    """
    meta_data_path = Path(config["meta_data_path"])
    meta_data = pd.read_csv(meta_data_path)
    game_meta_data = (
        meta_data[meta_data["GameName"] == game]
        .groupby("subject_id")["trial_id"]
        .apply(list)
        .reset_index()
    )
    return game_meta_data


def eye_gaze_to_webdataset(game: str, config: dict) -> None:
    """
    For each subject and trial in the specified game, reads the corresponding eye gaze data file
    and writes images to a WebDataset tar file.
    """
    webdataset_writer = WebDatasetWriter(config)

    raw_data_path = Path(config["raw_data_path"]) / game
    game_meta_data = get_game_meta_data(game, config)

    for _, row in game_meta_data.iterrows():
        subject_id = row["subject_id"]
        trial_ids = row["trial_id"]

        print(f"Reading data for subject: {subject_id}")

        for run, trial_id in enumerate(trial_ids):
            pattern = f"{trial_id}_{subject_id}*.txt"
            matched_files = list(raw_data_path.glob(pattern))

            # Create a tar file with subject_id and trail_id
            webdataset_writer.create_tar_file(
                file_name=f"{subject_id}_{trial_id}",
                write_path=config["processed_data_path"] + f"/{game}/",
            )

            if not matched_files:
                print(f"Warning: No files found for pattern '{pattern}'")
                continue

            file_path = matched_files[0]
            read_path = file_path.with_suffix("")  # remove '.txt'
            eye_gaze = read_gaze_data(file_path)

            # Write the game frames to .tar files
            tar_bz2_file = read_path.with_name(f"{read_path.name}.tar.bz2")
            extract_images_and_write_to_webdataset(
                tar_bz2_file, webdataset_writer, eye_gaze
            )

    # Close the webdataset writer gracefully
    webdataset_writer.close()
