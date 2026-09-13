import random
from datetime import datetime

from tqdm import tqdm

import tiktoken
import torch

from datasets import load_dataset
from safetensors.torch import save_file


class DataSource:

    def __init__(self, name, hf_id, hf_name, hf_split, item_field):
        self.name = name
        self.hf_id = hf_id
        self.hf_name = hf_name
        self.hf_split = hf_split
        self.item_field = item_field

        self.tokens_desired = 0
        self.tokens_used = 0
        self.iterator_restarts = 0

        self.restart_iterator()


    def restart_iterator(self):
        dataset = load_dataset(
            self.hf_id,
            name=self.hf_name,
            split=self.hf_split,
        )
        self.iterator = iter(dataset.shuffle(seed=random.randint(0, 1000)))
        self.iterator_restarts += 1


    def __next__(self):
        try:
            item = next(self.iterator)
        except StopIteration:
            self.restart_iterator()
            item = next(self.iterator)

        return item[self.item_field]



def log(s):
    print(f"{datetime.now()}: {s}")



def main():
    random.seed(42)
    total_tokens_desired = 10_000_000_000

    sources = [
        DataSource(
            name="FineWeb",
            hf_id="HuggingFaceFW/fineweb",
            hf_name="sample-10BT",
            hf_split="train",
            item_field="text",
        ),
        DataSource(
            name="FineWeb-Edu",
            hf_id="HuggingFaceFW/fineweb-edu",
            hf_name="sample-10BT",
            hf_split="train",
            item_field="text",
        ),
        DataSource(
            name="Simple English Wikipedia",
            hf_id="answerdotai/simplewiki",
            hf_name="articles",
            hf_split="train",
            item_field="md",
        ),
    ]

    ratios = {
        "FineWeb": 45,
        "FineWeb-Edu": 45,
        "Simple English Wikipedia": 10
    }
    total_ratios = sum(v for v in ratios.values())
    log("Generating dataset; per-source counts")
    for source in sources:
        adjusted_ratio = ratios[source.name] / total_ratios
        source.tokens_desired = int(total_tokens_desired * adjusted_ratio)
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
        print(source.name)
        print(text)
        import time
        time.sleep(2)

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
        log(f"{source.name}: {source.tokens_used:,d} / {source.tokens_desired:,d} ({ratio:.3f}, {source.iterator_restarts} restarts)")
    log(f"Total: {total_tokens_generated:,d}")

    log("Catting...")
    result = torch.cat(generated_tokens)
    generated_tokens = None
    log(f"Catted into a tensor of shape {result.shape}")

    log("Saving...")
    save_file({"tokens": result}, "./foo.safetensors")
    log("Saved")

    log("Done")


if __name__ == "__main__":
    main()
