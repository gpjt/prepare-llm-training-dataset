from pathlib import Path
import json
import tempfile

import click
import tiktoken

from huggingface_hub import snapshot_download
from safetensors.torch import load_file


@click.command()
@click.argument("run_dir")
def main(run_dir):
    run_dir = Path(run_dir)
    with open(run_dir / "conf.json") as f:
        conf = json.load(f)

    upload_dataset_name = conf['upload_dataset_name']

    with tempfile.TemporaryDirectory() as ds_dir:
        snapshot_download(
            upload_dataset_name,
            repo_type="dataset",
            local_dir=ds_dir,
            allow_patterns="*",
        )

        ds = load_file(Path(ds_dir) / "train.safetensors")

        tokens = ds["tokens"]
        print("\n" * 5)

    tokens_to_print = tokens[:10_000].tolist()

    tokenizer = tiktoken.get_encoding("gpt2")
    decoded = tokenizer.decode(tokens_to_print)
    print(decoded)


if __name__ == "__main__":
    main()
