import logging

import click
import cloudpickle
import polars as pl
from datasets import load_dataset

from src.encoder import LongTextEncoder


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


def parse_dataset(
    ds,
    encoder: LongTextEncoder,
    limit: int = 0,
    normalize: bool = False,
    combine_method: str = "weighted",
):
    """
    データセットをパースして埋め込みを計算する
    """

    # polars.DataFrame を取得
    result = ds.to_polars()

    # リミット処理(開発用)
    if limit > 0:
        result = result.head(limit)

    # id カラムを作成
    result = result.with_columns(
        pl.col("url")
        .str.replace(r".*/([0-9]*)/$", "$1")
        .cast(pl.Int32)
        .alias("id")
    )

    # 埋め込み対象のテキストを作成
    sentence = result.map_rows(make_sentence).select(
        pl.col("map").alias("sentence")
    )

    # 埋め込み計算
    embeddings = sentence.map_rows(
        lambda x: encoder.split_and_encode(
            x, normalize=normalize, combine_method=combine_method
        )
    ).rename(
        {
            "column_0": "embed",
            "column_1": "token_counts",
            "column_2": "total_tokens",
            "column_3": "char_counts",
            "column_4": "total_chars",
            "column_5": "sentence_chars",
        }
    )

    # 結果を結合
    result = pl.concat([result, sentence, embeddings], how="horizontal")

    return result


def make_dataset(
    dataset,
    model_name: str = "intfloat/multilingual-e5-small",
    limit: int = 0,
    chunk_overlap: int = 50,
    token_margin: int = 20,
    use_gpu: bool = True,
    normalize: bool = False,
    combine_method: str = "weighted",
):
    """
    Train、Validation、Test のデータを作る。辞書形式で返す。
    """

    # logger 初期化
    logger = logging.getLogger(__name__)

    # 埋め込みモデルを初期化
    encoder = LongTextEncoder(
        model_name=model_name,
        chunk_overlap=chunk_overlap,
        token_margin=token_margin,
        use_gpu=use_gpu,
    )

    # 結果を返す変数を初期化
    results = {}

    # データセットのキー毎に処理
    for key in dataset:

        # 進捗表示
        logger.info(f"==== {key}")

        # パースと埋め込み計算
        results[key] = parse_dataset(
            dataset[key],
            encoder,
            limit=limit,
            normalize=normalize,
            combine_method=combine_method,
        )

        # デバッグ出力
        results[key].write_excel(f"data/debug_{key}.xlsx")

    return results


@click.command()
@click.argument("output_filepath", type=click.Path())
@click.option(
    "--model_name", type=str, default="intfloat/multilingual-e5-small"
)
@click.option("--chunk_overlap", type=int, default=50)
@click.option("--token_margin", type=int, default=20)
@click.option("--use_gpu", type=bool, default=False)
@click.option("--limit", type=int, default=0)
@click.option("--normalize", type=bool, default=False)
@click.option(
    "--combie_method", type=str, default="weighted", help="weighted or mean"
)
def main(**kwargs):

    # load raw dataset
    raw_dataset = load_dataset(
        "shunk031/livedoor-news-corpus", trust_remote_code=True
    )

    # make dataset
    dataset = make_dataset(
        raw_dataset,
        model_name=kwargs["model_name"],
        chunk_overlap=kwargs["chunk_overlap"],
        token_margin=kwargs["token_margin"],
        use_gpu=kwargs["use_gpu"],
        limit=kwargs["limit"],
        normalize=kwargs["normalize"],
        combine_method=kwargs["combie_method"],
    )

    # output to cloudpickle
    cloudpickle.dump(dataset, open(kwargs["output_filepath"], "wb"))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
