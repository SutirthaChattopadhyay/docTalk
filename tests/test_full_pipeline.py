from pdf_parser import parse_pdf
from embeddings import EmbeddingPipeline

pdf_path = r"C:\Users\sutir\Downloads\public.pdf"

# 1. Extract and chunk the PDF
print("Parsing PDF...")
chunks = parse_pdf(pdf_path)

print(f"Total chunks created: {len(chunks)}")

# 2. Build FAISS index
print("\nBuilding embedding index...")
pipeline = EmbeddingPipeline()
pipeline.build(chunks)

# 3. Ask a question
question = "What is the haemoglobin level?"

print(f"\nQuestion: {question}")

results = pipeline.search(question, top_k=3)

# 4. Display retrieved chunks
print("\nRetrieved results:")

for result in results:
    print("-" * 50)
    print("Page:", result["page"])
    print("Method:", result["method"])
    print("Score:", round(result["score"], 3))
    print("Text:", result["text"])