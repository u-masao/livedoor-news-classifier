import logging
from typing import Any, Dict

import click
import cloudpickle
import qdrant_client
from qdrant_client.http import models


def init_qdrant(
    dataset: Dict[str, Any],
    qdrant_port: int,
    qdrant_host: str,
    qdrant_collection: str,
):

    # データを取得
    data = dataset["train"]

    # 埋め込みの次元を取得
    dimension = data["embed"][0].shape[0]

    # init qdrant client
    client = qdrant_client.QdrantClient(qdrant_host, port=qdrant_port)

    # delete qdrant collection
    client.delete_collection(collection_name=qdrant_collection)

    # init qdrant collection
    client.create_collection(
        collection_name=qdrant_collection,
        vectors_config=models.VectorParams(
            size=dimension, distance=models.Distance.COSINE
        ),
    )

    # upload data
    client.upload_collection(
        collection_name=qdrant_collection,
        ids=data["id"],
        vectors=data["embed"].to_list(),
        payload=data[
            [
                "id",
                "title",
                "category",
                "sentence",
            ]
        ].to_dicts(),
        parallel=4,
        max_retries=3,
    )

    # make index
    client.create_payload_index(
        collection_name=qdrant_collection,
        field_name="category",
        field_schema="integer",
    )


@click.command()
@click.argument("input_filepath", type=click.Path())
@click.option("--qdrant_port", type=int, default=6333)
@click.option("--qdrant_host", type=str, default="127.0.0.1")
@click.option("--qdrant_collection", type=str, default="livedoor_news")
def main(**kwargs):

    # load dataset
    dataset = cloudpickle.load(open(kwargs["input_filepath"], "rb"))

    # init qdrant
    init_qdrant(
        dataset,
        qdrant_port=kwargs["qdrant_port"],
        qdrant_host=kwargs["qdrant_host"],
        qdrant_collection=kwargs["qdrant_collection"],
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
