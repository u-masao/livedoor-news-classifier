import logging

import click
import cloudpickle
from datasets import load_dataset
import polars as pl
import textwrap

# import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from langchain_text_splitters import SentenceTransformersTokenTextSplitter


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


chunk_overlap=50
model_name = 'intfloat/multilingual-e5-small'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
splitter = SentenceTransformersTokenTextSplitter(model_name=model_name,chunk_overlap=chunk_overlap)


def make_sentence(row):
    title = row[2]
    body=row[3]
    sentence = f'''
        ### タイトル

        {title}

        ### 本文

        {body}
    '''
    sentence= textwrap.dedent(sentence)
    return sentence

def embed(sentence):

    # Tokenize the input texts
    batch_dict = tokenizer(sentence, max_length=512, padding=True, truncation=True, return_tensors='pt')
    
    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    
    # normalize embeddings
    # embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings

def split_and_embed(row):
    sentence = row[5]
    text_chunks = splitter.split_text(text=sentence)
    embeddings  = embed(text_chunks)
    return embeddings

def parse_dataset(ds):
    result = ds.to_polars()
    result = result.with_columns(pl.col('url').str.replace(
        r'.*/([0-9]*)/$','$1').cast(pl.Int32).alias('id'))
    sentence = result.map_rows(make_sentence).select(pl.col('map').alias('sentence'))
    result = result.with_columns(sentence)
    embeddings  = result.map_rows(split_and_embed).select(pl.col('map').alias('embed')
    print(embeddings)
    result = result.with_columns(embeddings)
    result.write_excel('data/temp.xlsx')
    return result

def make_dataset(dataset):
    logger = logging.getLogger(__name__)
    results = {}
    for key in dataset:
        logger.info(f"==== {key}")
        results[key] = parse_dataset(dataset[key])

    logger.info(results)

    return results


@click.command()
@click.argument("output_filepath", type=click.Path())
def main(**kwargs):

    raw_dataset = load_dataset("shunk031/livedoor-news-corpus")
    dataset = make_dataset(raw_dataset)
    cloudpickle.dump(dataset, open(kwargs["output_filepath"], "wb"))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
