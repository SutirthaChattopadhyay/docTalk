from embeddings import EmbeddingPipeline

chunks = [
    {
        "chunk_id": 0,
        "page": 1,
        "text": "Haemoglobin level is 15 g/dL. Normal range is 13 to 17 g/dL.",
        "method": "tesseract"
    },
    {
        "chunk_id": 1,
        "page": 1,
        "text": "Total leukocyte count is 5100 cells per microlitre.",
        "method": "tesseract"
    },
    {
        "chunk_id": 2,
        "page": 2,
        "text": "Platelet count is 250000 per microlitre.",
        "method": "pdfplumber"
    }
]

pipeline = EmbeddingPipeline()
pipeline.build(chunks)

results = pipeline.search(
    "What is the haemoglobin level?",
    top_k=3
)

print("\nSearch Results:")
for result in results:
    print(result)


pipeline.save("test_index")

loaded_pipeline = EmbeddingPipeline()
loaded_pipeline.load("test_index")

results = loaded_pipeline.search(
    "What is the haemoglobin level?",
    top_k=1
)

print("\nResults after loading:")
print(results[0])
