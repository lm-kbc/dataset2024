import argparse
import json
from pathlib import Path

import yaml
from loguru import logger

from models.user_config import Models


def main():
    parser = argparse.ArgumentParser(description="Run Baseline Models")

    parser.add_argument(
        "-c", "--config_file",
        type=str,
        required=True,
        help="Path to the configuration file"
    )
    parser.add_argument(
        "-i", "--input_file",
        type=str,
        required=True,
        help="Path to the input file"
    )
    parser.add_argument(
        "-o", "--output_file",
        type=str,
        required=False,
        help="Path to the output file"
    )

    args = parser.parse_args()

    # Load the configuration file
    logger.info(f"Loading the YAML configuration file `{args.config_file}`...")
    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    # File paths
    input_file = args.input_file
    output_file = args.output_file
    if not output_file:
        output_dir = Path(__file__).resolve().parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        file_name = Path(args.config_file).stem
        output_file = output_dir / f"{file_name}.jsonl"

    # Load the input file
    logger.info(f"Loading the input file `{input_file}`...")
    with open(input_file) as f:
        input_rows = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(input_rows):,} rows.")

    # Load the model
    m = Models.get_model(config["model"])
    model = m(config)

    # Generate predictions
    results = model.generate_predictions(input_rows)

    # Save the results
    logger.info(f"Saving the results to `{output_file}`...")
    with open(output_file, "w+") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    logger.info("Done!")


if __name__ == "__main__":
    main()
