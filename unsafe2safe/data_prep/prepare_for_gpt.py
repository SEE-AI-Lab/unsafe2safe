import json
from argparse import ArgumentParser

from unsafe2safe.data_prep.generate_txt_dataset import DELIMITER_0, DELIMITER_1, STOP


def main(input_path: str, output_path: str):
    with open(input_path) as f:
        prompts = [json.loads(line) for line in f]

    with open(output_path, "w") as f:
        for prompt in prompts:
            prompt_for_gpt = {
                "prompt": f"{prompt['input']}{DELIMITER_0}",
                "completion": f"{prompt['edit']}{DELIMITER_1}{prompt['output']}{STOP}",
            }
            f.write(f"{json.dumps(prompt_for_gpt)}\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input-path", required=True, type=str)
    parser.add_argument("--output-path", required=True, type=str)
    args = parser.parse_args()
    main(args.input_path, args.output_path)
