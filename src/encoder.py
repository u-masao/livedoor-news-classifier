import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


class LongTextEncoder:
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        chunk_overlap: int = 50,
        token_margin: int = 20,
        use_gpu: bool = True,
    ):

        self.chunk_overlap = chunk_overlap
        self.token_margin = token_margin
        self.model_name = model_name

        self.device = "cpu"
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda:0"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.splitter = SentenceTransformersTokenTextSplitter(
            model_name=self.model_name,
            chunk_overlap=self.chunk_overlap,
            tokens_per_chunk=self.tokenizer.model_max_length
            - self.token_margin,
        )

    @staticmethod
    def average_pool(
        last_hidden_states: Tensor, attention_mask: Tensor
    ) -> Tensor:
        """
        アベレージプーリング
        """
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    @torch.no_grad()
    def encode(self, sentences: List[str], normalize: bool = False):
        """
        埋め込み表現を得る
        """

        # Tokenize the input texts
        batch_dict = self.tokenizer(
            sentences,
            max_length=self.tokenizer.model_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # embedding
        outputs = self.model(**batch_dict)

        # average pooling
        embeddings = LongTextEncoder.average_pool(
            outputs.last_hidden_state, batch_dict["attention_mask"]
        )

        # normalize embeddings
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def split_and_embed(self, row: Tuple[str], prefix: str = "passage: "):
        """
        センテンスの分割と埋め込み処理
        """
        # init logger
        logger = logging.getLogger(__name__)

        # pickup target data
        sentence = row[0]

        # split
        text_chunks = self.splitter.split_text(text=sentence)

        # add prefix
        text_chunks = [f"{prefix}{x}" for x in text_chunks]

        # encode
        embeddings = torch.flatten(self.encode(text_chunks)).tolist()

        metrics = self.count_tokens(sentence, text_chunks)
        logger.info(metrics)
        return [embeddings] + list(metrics.values())

    def count_tokens(self, sentence: str, text_chunks: List[str]):
        # tokenize for token counts
        batch_ids = self.tokenizer(
            text_chunks,
            max_length=self.tokenizer.model_max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # count tokens and chars
        sentence_chars: int = len(sentence)
        token_counts: List[int] = (
            np.array(batch_ids["attention_mask"]).sum(axis=1).tolist()
        )
        total_tokens: int = np.array(batch_ids["attention_mask"]).sum().item()
        char_counts: List[int] = [len(x) for x in text_chunks]
        total_chars: int = sum(char_counts)
        metrics = {
            "token_counts": token_counts,
            "total_tokens": total_tokens,
            "char_counts": char_counts,
            "total_chars": total_chars,
            "sentence_chars": sentence_chars,
        }
        return metrics
