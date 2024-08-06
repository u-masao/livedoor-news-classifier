import logging

import click
import cloudpickle
import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from datasets import load_dataset
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """
    アベレージプーリング
    """
    last_hidden = last_hidden_states.masked_fill(
        ~attention_mask[..., None].bool(), 0.0
    )
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


chunk_overlap = 50
prefix_tokens = 10
model_name = "intfloat/multilingual-e5-small"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"{device=}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
splitter = SentenceTransformersTokenTextSplitter(
    model_name=model_name,
    chunk_overlap=chunk_overlap,
    tokens_per_chunk=tokenizer.model_max_length - prefix_tokens,
)


def make_sentence(row):
    """
    センテンスを組み立てる
    """
    sentence = ""
    sentence += "### タイトル\n\n"
    sentence += row[2]  # title
    sentence += "\n### 本文\n\n"
    sentence += row[3]  # body
    return sentence


@torch.no_grad()
def embed(sentences, normalize=False):
    """
    埋め込み表現を得る
    """

    # Tokenize the input texts
    batch_dict = tokenizer(
        sentences,
        # max_length=512,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)

    outputs = model(**batch_dict)
    embeddings = average_pool(
        outputs.last_hidden_state, batch_dict["attention_mask"]
    )

    # normalize embeddings
    if normalize:
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings


def split_and_embed(row):
    """
    センテンスの分割と埋め込み処理
    """
    sentence = row[0]
    sentence_chars = len(sentence)
    logger = logging.getLogger(__name__)
    text_chunks = splitter.split_text(text=sentence)
    text_chunks = [f"passage: {x}" for x in text_chunks]

    batch_ids = tokenizer(
        text_chunks,
        max_length=tokenizer.model_max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    token_counts = np.array(batch_ids["attention_mask"]).sum(axis=1).tolist()
    total_tokens = np.array(batch_ids["attention_mask"]).sum().item()
    char_counts = [len(x) for x in text_chunks]
    total_chars = sum(char_counts)
    metrics = {
        "token_counts": token_counts,
        "total_tokens": total_tokens,
        "char_counts": char_counts,
        "total_chars": total_chars,
        "sentence_chars": sentence_chars,
    }
    logger.info(metrics)
    embeddings = torch.flatten(embed(text_chunks)).tolist()
    return (embeddings, total_chars, total_tokens, sentence_chars)


def parse_dataset(ds, limit: int = 0):
    """
    データセットをパースして埋め込みを計算する
    """

    # polars.DataFrame を取得
    result = ds.to_polars()

    # リミット処理(開発用)
    if limit > 0:
        result = result.head(limit)

    result = result.with_columns(
        pl.col("url")
        .str.replace(r".*/([0-9]*)/$", "$1")
        .cast(pl.Int32)
        .alias("id")
    )
    sentence = result.map_rows(make_sentence).select(
        pl.col("map").alias("sentence")
    )
    embeddings = sentence.map_rows(split_and_embed).rename(
        {
            "column_0": "embed",
            "column_1": "total_chars",
            "column_2": "total_tokens",
            "column_3": "sentence_chars",
        }
    )
    result = pl.concat([result, sentence, embeddings], how="horizontal")
    return result


def make_dataset(dataset, limit: int = 0):
    """
    Train、Validation、Test のデータを作る。辞書形式で返す。
    """

    # logger 初期化
    logger = logging.getLogger(__name__)

    # 結果を返す変数を初期化
    results = {}

    for key in dataset:
        # 進捗表示
        logger.info(f"==== {key}")

        # パースと埋め込み計算
        results[key] = parse_dataset(dataset[key], limit=limit)

        # デバッグ出力
        results[key].write_excel(f"data/debug_{key}.xlsx")

    return results


@click.command()
@click.argument("output_filepath", type=click.Path())
@click.option("--limit", type=int, default=0)
def main(**kwargs):

    # load raw dataset
    raw_dataset = load_dataset(
        "shunk031/livedoor-news-corpus", trust_remote_code=True
    )

    # make dataset
    dataset = make_dataset(raw_dataset, limit=kwargs["limit"])

    # output to cloudpickle
    cloudpickle.dump(dataset, open(kwargs["output_filepath"], "wb"))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
