import gradio as gr
import polars as pl
import qdrant_client
from sentence_transformers import SentenceTransformer

# init model
model_name = "intfloat/multilingual-e5-small"
model = SentenceTransformer(model_name)
print(f"model initialized: {model_name}")


def search(sentence):
    global model
    qdrant_port = 6333
    qdrant_host = "127.0.0.1"
    qdrant_collection = "livedoor_news"

    client = qdrant_client.QdrantClient(qdrant_host, port=qdrant_port)
    embedding = model.encode(f"query: {sentence}", normalize_embeddings=True)

    # init qdrant client
    client = qdrant_client.QdrantClient(qdrant_host, port=qdrant_port)

    print(embedding)

    search_result = client.search(
        collection_name=qdrant_collection,
        limit=10,
        query_vector=embedding,
        with_vectors=True,
        with_payload=True,
    )
    print(pl.DataFrame(search_result))
    return str(pl.DataFrame(search_result)["payload"])


with gr.Blocks() as demo:
    sentence = gr.Textbox(label="Search Keyword")
    output = gr.Markdown(label="Output")
    search_button = gr.Button("Search")
    search_button.click(
        fn=search, inputs=sentence, outputs=output, api_name="news search"
    )

if __name__ == "__main__":
    demo.launch(share=False, debug=True)
