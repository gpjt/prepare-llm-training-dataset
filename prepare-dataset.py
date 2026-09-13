import json
import random
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

import click
import tiktoken
import torch

from datasets import load_dataset
from huggingface_hub import HfApi
from safetensors.torch import save_file


class DataSource:

    def __init__(self, name, hf_id, hf_name, hf_split, item_field, weight):
        self.name = name
        self.hf_id = hf_id
        self.hf_name = hf_name
        self.hf_split = hf_split
        self.item_field = item_field
        self.weight = weight

        self.tokens_desired = 0
        self.tokens_used = 0
        self.iterators_used = 0

        self.restart_iterator()


    def restart_iterator(self):
        dataset = load_dataset(
            self.hf_id,
            name=self.hf_name,
            split=self.hf_split,
        )
        self.iterator = iter(dataset.shuffle(seed=random.randint(0, 1000)))
        self.iterators_used += 1


    def __next__(self):
        try:
            item = next(self.iterator)
        except StopIteration:
            self.restart_iterator()
            item = next(self.iterator)

        return item[self.item_field]



def log(s):
    print(f"{datetime.now()}: {s}")



@click.command()
@click.argument("run_dir")
def main(run_dir):
    run_dir = Path(run_dir)
    with open(run_dir / "conf.json") as f:
        conf = json.load(f)

    random.seed(conf["seed"])
    total_tokens_desired = conf["tokens_desired"]

    sources = [
        DataSource(**source) for source in conf["sources"]
    ]

    total_weights = sum(s.weight for s in sources)
    log("Generating dataset; per-source counts")
    for source in sources:
        ratio = source.weight / total_weights
        source.tokens_desired = int(total_tokens_desired * ratio)
        log(f"{source.name}: {source.tokens_desired:,d}")

    tqdms = {}
    for source in sources:
        tqdms[source.name] = tqdm(
            desc=source.name,
            total=source.tokens_desired,
            unit="token"
        )

    tokenizer = tiktoken.get_encoding("gpt2")
    total_tokens_generated = 0
    generated_tokens = []
    while total_tokens_generated < total_tokens_desired:
        least_tapped_source = None
        least_tapped_ratio = None
        for source in sources:
            ratio = source.tokens_used / source.tokens_desired
            if least_tapped_ratio is None or ratio < least_tapped_ratio:
                least_tapped_source = source
                least_tapped_ratio = ratio

        source = least_tapped_source

        text = next(source)

        tokens = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
        tokens.append(tokenizer.eot_token)

        token_count = len(tokens)
        source.tokens_used += token_count
        tqdms[source.name].update(token_count)
        total_tokens_generated += token_count

        generated_tokens.append(
            torch.tensor(tokens, dtype=torch.uint16)
        )

    for t in tqdms.values():
        t.close()

    log("\n\n\nDone generating tokens")
    for source in sources:
        ratio = source.tokens_used / source.tokens_desired
        log(f"{source.name}: {source.tokens_used:,d} / {source.tokens_desired:,d} ({ratio:.3f}, {source.iterators_used} iterators)")
    log(f"Total: {total_tokens_generated:,d}")

    log("Catting...")
    result = torch.cat(generated_tokens)
    generated_tokens = None
    log(f"Catted into a tensor of shape {result.shape}")

    log("Saving...")
    safetensors_file = run_dir / "train.safetensors"
    save_file({"tokens": result}, safetensors_file)
    log("Saved")

    upload_dataset_name = conf['upload_dataset_name']
    log("Uploading to {upload_dataset_name}")
    api = HfApi()
    api.create_repo(
        repo_id=upload_dataset_name,
        repo_type="dataset",
        exist_ok=True,
    )
    api.upload_file(
        path_or_fileobj=safetensors_file,
        path_in_repo=safetensors_file.name,
        repo_id=upload_dataset_name,
        repo_type="dataset"
    )

    log("Done")


if __name__ == "__main__":
    main()
