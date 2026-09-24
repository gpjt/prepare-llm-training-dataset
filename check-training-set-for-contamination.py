from pathlib import Path
import tempfile

from tqdm import tqdm
import click
import tiktoken

from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from dataset_contamination_helpers import hash_doc


@click.command()
@click.argument("dataset")
@click.argument("split")
@click.argument("forbidden_hashes_input_file")
def main(dataset, split, forbidden_hashes_input_file):
    split_filename = f"{split}.safetensors"
    with tempfile.TemporaryDirectory() as ds_dir:
        snapshot_download(
            dataset,
            repo_type="dataset",
            local_dir=ds_dir,
            allow_patterns=split_filename,
        )

        ds = load_file(Path(ds_dir) / split_filename)

        tokens = ds["tokens"]
        print("\n" * 5)

    forbidden = {}
    with open(forbidden_hashes_input_file, "r") as f:
        while True:
            line = f.readline()
            if line == "":
                break
            digest, tok_count_s = line.split(" ")
            tok_count = int(tok_count_s)
            forbidden[digest] = tok_count

    tokenizer = tiktoken.get_encoding("gpt2")
    this_doc = []
    contaminations = set()
    print("Checking...")
    for tok in tqdm(tokens.numpy()):
        if tok == tokenizer.eot_token:
            digest = hash_doc(this_doc)
            if digest in forbidden:
                contaminations.add((digest, forbidden[digest]))
            this_doc = []
        else:
            this_doc.append(tok)

    if len(contaminations) == 0:
        print("DATASET CLEAN")
    else:
        print("CONTAMINATED\n\nFound:")
        contaminated_tokens = 0
        for digest, tokens in contaminations:
            contaminated_tokens += tokens
        total_forbidden_tokens = sum(forbidden.values())
        percent_of_forbidden = (contaminated_tokens * 100) / total_forbidden_tokens
        print(f"Total contamination is {contaminated_tokens} out of {total_forbidden_tokens} tokens ({percent_of_forbidden:.2f}%)")



if __name__ == "__main__":
    main()
