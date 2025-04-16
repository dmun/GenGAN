import os
import json
import soundfile as sf
from pathlib import Path


def generate_preprocess_json(dataset_path, output_path, set_name="train"):
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    preprocess_data = []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".flac"):
                file_path = Path(root) / file
                try:
                    with sf.SoundFile(file_path) as audio_file:
                        duration = len(audio_file) / audio_file.samplerate
                    preprocess_data.append(
                        [str(file_path.resolve()), 0.0, round(duration, 2)]
                    )
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    preprocess_json_path = output_path / f"libri_{set_name}_preprocess.json"
    with open(preprocess_json_path, "w") as f:
        json.dump(preprocess_data, f, indent=2)

    print(f"Generated {preprocess_json_path}")


# Example usage:
generate_preprocess_json("./train-clean-100", "./data_files", "train")
generate_preprocess_json("./test-clean", "./data_files", "test")
