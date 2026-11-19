
from getpass import getpass
from rag_pipeline import DocTalkRAG

PDF_PATH = r"C:\Users\sutir\Downloads\public.pdf"

api_key = getpass("Gemini API key: ")

# Initialize DocTalk
doctalk = DocTalkRAG(api_key=api_key)

# Load document
info = doctalk.load_document(PDF_PATH)

print("\nDocument information:")
print(info)

# Ask a question
while True:
    question = input("\nAsk DocTalk (type exit to quit): ")

    if question.lower() == "exit":
        break

    try:
        result = doctalk.ask(question)

        print("\nDocTalk:")
        print(result["answer"])

        print("\nSources:")
        for source in result["sources"]:
            print(
                f"Page {source['page']} | "
                f"Score: {source['score']:.3f}"
            )

    except Exception as e:
        print("Error:", e)
