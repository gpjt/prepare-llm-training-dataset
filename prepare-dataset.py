from tqdm import tqdm

import tiktoken
import torch

from datasets import load_dataset


class DataSource:

    def __init__(self, name, hf_id, hf_name, hf_split, item_field):
        self.name = name
        self.hf_id = hf_id
        self.hf_name = hf_name
        self.hf_split = hf_split
        self.item_field = item_field

        self.tokens_desired = 0
        self.tokens_used = 0

        self.create_iterator()


    def create_iterator(self):
        dataset = load_dataset(
            self.hf_id,
            name=self.hf_name,
            split=self.hf_split,
            streaming=True
        )
        self.iterator = iter(dataset)


    def __next__(self):
        try:
            item = next(self.iterator)
        except StopIteration:
            self.create_iterator()
            item = next(self.iterator)

        return item[self.item_field]



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




def main():
    total_tokens_desired = 10_000_000_000

    ratios = {
        "FineWeb": 45,
        "FineWeb-Edu": 45,
        "Simple English Wikipedia": 10
    }
    total_ratios = sum(v for v in ratios.values())
    print("Generating dataset; per-source counts")
    for source in sources:
        adjusted_ratio = ratios[source.name] / total_ratios
        source.tokens_desired = int(total_tokens_desired * adjusted_ratio)
        print(f"{source.name}: {source.tokens_desired:,d}")

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
        print(source.name)

        text = next(source)
        print(repr(text))

        tokens = tokenizer.encode(text)
        tokens.append(tokenizer.eot_token)

        token_count = len(tokens)
        source.tokens_used += token_count
        tqdms[source.name].update(token_count)
        total_tokens_generated += token_count

        generated_tokens.append(
            torch.tensor(tokens, dtype=torch.uint16)
        )

    print("\n\n\nDone generating tokens")
    for source in sources:
        ratio = source.tokens_used / source.tokens_desired
        print(f"{source.name}: {source.tokens_used} / {source.tokens_desired} ({ratio:.3f})")
    print(f"Total: {total_tokens_generated}")


    print("done")


if __name__ == "__main__":
    main()
