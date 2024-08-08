import gradio as gr
import polars as pl
import qdrant_client
from sentence_transformers import SentenceTransformer

# init model
model_name = "intfloat/multilingual-e5-small"
model = SentenceTransformer(model_name)
_ = model.encode("dummy string")
print(f"model initialized: {model_name}")

# init qdrant client
qdrant_port = 6333
qdrant_host = "127.0.0.1"
qdrant_collection = "livedoor_news"
client = qdrant_client.QdrantClient(qdrant_host, port=qdrant_port)


def search(sentence: str, limit: int):
    global model, client

    # encode
    embedding = model.encode(f"query: {sentence}", normalize_embeddings=True)

    # init qdrant client
    client = qdrant_client.QdrantClient(qdrant_host, port=qdrant_port)

    # search
    scored_results = client.search(
        collection_name=qdrant_collection,
        limit=limit,
        query_vector=embedding,
        with_vectors=False,
        with_payload=True,
    )

    # parse results
    results = []
    for scored_result in scored_results:
        scored_result_dict = scored_result.model_dump()
        payload = scored_result_dict.pop("payload")
        payload.pop("sentence")
        result = {}
        result.update(payload)
        result["score"] = scored_result_dict["score"]
        results.append(result)

    # make markdown
    return (
        f"found: {len(scored_results)}\n\n",
        pl.DataFrame(results).to_pandas(),
    )


# define UI widgets
with gr.Blocks() as demo:
    sentence = gr.Textbox(label="Search Keyword")
    limit = gr.Number(label="Search limit", value=5, maximum=1000, minimum=1)
    search_button = gr.Button("Search")
    output = gr.Markdown(label="Output")
    output_dataframe = gr.Dataframe(label="Output dataframe")
    search_button.click(
        fn=search,
        inputs=[sentence, limit],
        outputs=[output, output_dataframe],
        api_name="news search",
    )

if __name__ == "__main__":
    demo.launch(share=False, debug=True)
