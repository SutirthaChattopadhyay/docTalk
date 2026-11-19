
from google import genai


class AnswerGenerator:
    """Generate answers using Gemini and retrieved document context."""

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-3.5-flash-lite"

    def generate(self, question: str, context: str) -> str:
        """
        Generate an answer grounded in the supplied document context.

        Raises:
            RuntimeError: If Gemini fails to generate an answer.
        """

        prompt = f"""
        You are DocTalk, a document question-answering assistant.

        Answer the user's question using ONLY the provided document context.

        Rules:
        1. Do not invent or assume information.
        2. If the answer is not in the context, say it was not found.
        3. Preserve medical values, units and medication names exactly.
        4. Do not interpret OCR errors as verified medical facts.
        5. Flag unclear or potentially inaccurate OCR text.
        6. Do not provide medical diagnoses.
        7. Keep answers clear and concise.

        DOCUMENT CONTEXT:
        {context}

        USER QUESTION:
        {question}

        ANSWER:
        """

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )

            answer = response.text

            if not answer:
                raise RuntimeError("Gemini returned an empty response.")

            return answer

        except Exception as e:
            print(f"Gemini API error: {e}")

            raise RuntimeError(
                "DocTalk could not generate an answer. "
                "Please try again later."
            ) from e
