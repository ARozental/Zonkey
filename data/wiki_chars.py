# Import from the external HuggingFace datasets library
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset
from torch.utils.data import Dataset
from configs.default_config import Config
from functools import partial
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'


def load_wikipedia(max_examples: int | None = None):
    if max_examples is None:
        split = "train[:95%]"
    else:
        split = f"train[:{max_examples}]"
    wiki_dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split=split)
    return wiki_dataset


def collate_char(batch):
    return {
        "full_texts": torch.stack(batch, dim=0)
    }


class WikipediaCharsDataset(Dataset):
    def __init__(self, max_examples: int | None = None):
        self.wiki_dataset = load_wikipedia(max_examples=max_examples)

    def __len__(self):
        return len(self.wiki_dataset)

    def __getitem__(self, idx: int):
        article = self.wiki_dataset[idx]
        text = article["text"]

        # Model real UTF-8 bytes. ord(char) % 256 aliases unrelated Unicode
        # codepoints and is not the byte-level representation used by H-Net.
        tokens = list(text.encode("utf-8"))
        max_length = Config.MAX_DOC_LENGTHS[0]

        # Do not train almost exclusively on encyclopedic lead paragraphs. Preserve
        # some starts for document-opening structure, but sample most windows from
        # throughout long articles.
        if len(tokens) > max_length:
            if torch.rand(()) < 0.2:
                start = 0
            else:
                start = int(torch.randint(0, len(tokens) - max_length + 1, ()).item())
            tokens = tokens[start:start + max_length]

        if len(tokens) == Config.MAX_DOC_LENGTHS[0] - 1:
            tokens.append(1)
        elif len(tokens) < Config.MAX_DOC_LENGTHS[0]:
            padding = [1] + [0] * (Config.MAX_DOC_LENGTHS[0] - len(tokens) - 1)
            tokens.extend(padding)
        
        return torch.tensor(tokens, dtype=torch.long)

def create_dataloader(
    batch_size: int = 3,
    max_examples: int | None = None,
    num_workers: int = 0,
    shuffle: bool = True
) -> torch.utils.data.DataLoader:
    dataset = WikipediaCharsDataset(max_examples=max_examples)

    collate_fn = partial(collate_char)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=4 if num_workers > 0 else None,
    )
