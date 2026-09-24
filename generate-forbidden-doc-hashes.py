from pathlib import Path
import tempfile

from tqdm import tqdm
import click
import tiktoken

from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from dataset_contamination_helpers import hash_doc


@click.command()
@click.argument("forbidden_dataset")
@click.argument("forbidden_split")
@click.argument("forbidden_hashes_output_file")
def main(forbidden_dataset, forbidden_split, forbidden_hashes_output_file):
    split_filename = f"{forbidden_split}.safetensors"
    with tempfile.TemporaryDirectory() as ds_dir:
        snapshot_download(
            forbidden_dataset,
            repo_type="dataset",
            local_dir=ds_dir,
            allow_patterns=split_filename,
        )

        ds = load_file(Path(ds_dir) / split_filename)

        tokens = ds["tokens"]
        print("\n" * 5)

    tokenizer = tiktoken.get_encoding("gpt2")
    this_doc = []
    print("Generating hash file...")
    with open(forbidden_hashes_output_file, "w") as f:
        for tok in tqdm(tokens.numpy()):
            if tok == tokenizer.eot_token:
                if len(this_doc) < 10:
                    raise Exception(f"Unexpectedly short doc: {tokenizer.decode(this_doc)!r}")

                f.write(f"{hash_doc(this_doc)} {len(this_doc)}\n")
                this_doc = []
            else:
                this_doc.append(tok)


if __name__ == "__main__":
    main()
