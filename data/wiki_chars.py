# Import from the external HuggingFace datasets library
import torch
import sys
import os
import unicodedata

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import load_dataset
from torch.utils.data import Dataset
from configs.default_config import Config
from functools import partial
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'


# 0 = PAD, 1 = END (appended after the text), 2 = "foreign character".
UNK_ID = 2
_FIXES = {"‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"',
          "–": "-", "—": "-", "―": "-", "−": "-", "‐": "-", "‑": "-", "…": "...",
          "ł": "l", "Ł": "L", "đ": "d", "Đ": "D", "ı": "i", "œ": "oe", "Œ": "OE"}


def text_to_ids(text: str) -> list[int]:
    """The only text -> id mapping. Keeps Latin-1 as is, strips accents outside it
    (š->s, ā->a, ō->o), maps typographic punctuation to ASCII, emits one UNK per run of
    anything else, and never emits 0 (PAD) or 1 (END). Every id is in [2, 255], so the
    256-entry embedding and the chr(i % 256) decode are unchanged."""
    ids = []
    for ch in text:
        ch = _FIXES.get(ch, ch)
        if len(ch) == 1 and 2 <= ord(ch) < 256:
            ids.append(ord(ch))
            continue
        base = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
        if not base:
            continue                            # bare combining mark (stress accents): drop
        if all(2 <= ord(c) < 256 for c in base):
            ids.extend(ord(c) for c in base)
        elif not ids or ids[-1] != UNK_ID:
            ids.append(UNK_ID)
    return ids


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
        
        tokens = text_to_ids(text)
        
        if len(tokens) == Config.MAX_DOC_LENGTHS[0] - 1:
            tokens.append(1)
        elif len(tokens) < Config.MAX_DOC_LENGTHS[0]:
            padding = [1] + [0] * (Config.MAX_DOC_LENGTHS[0] - len(tokens) - 1)
            tokens.extend(padding)
        elif len(tokens) > Config.MAX_DOC_LENGTHS[0]:
            tokens = tokens[:Config.MAX_DOC_LENGTHS[0]]
        
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
