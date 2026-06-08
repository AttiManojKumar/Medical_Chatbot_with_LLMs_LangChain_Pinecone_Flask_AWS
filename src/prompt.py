SYSTEM_PROMPT = """
You are a helpful medical assistant for answering questions about medical topics.

Use only the retrieved information provided to you.
If the answer is not present in the retrieved information, say you do not know.
Keep the answer concise and easy to understand.
"""


def build_gemini_prompt(question, docs):
    """Build the prompt that Gemini receives after Pinecone retrieval."""
    context = "\n\n".join(doc.page_content for doc in docs)

    return f"""
{SYSTEM_PROMPT}

Retrieved information:
{context}

Question:
{question}
"""
