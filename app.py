from flask import Flask, render_template, request


app = Flask(__name__)

qa_context = {
    "ready": False,
    "initializing": False,
    "error": None,
    "docsearch": None,
    "gemini_client": None,
}


def initialize_chatbot() -> None:
    """Connect the Flask app to Pinecone retrieval and Gemini answering."""
    if qa_context["ready"]:
        return

    from src.helper import (
        INDEX_NAME,
        build_vector_store,
        create_or_get_pinecone_index,
        download_hugging_face_embeddings,
        get_gemini_client,
        get_pinecone_client,
        load_api_keys,
        require_key,
    )

    qa_context["initializing"] = True
    keys = load_api_keys()
    pinecone_api_key = require_key(keys, "pinecone")
    gemini_api_key = require_key(keys, "gemini")

    embeddings = download_hugging_face_embeddings()
    pc = get_pinecone_client(pinecone_api_key)
    create_or_get_pinecone_index(pc, INDEX_NAME)

    qa_context["docsearch"] = build_vector_store(INDEX_NAME, embeddings)
    qa_context["gemini_client"] = get_gemini_client(gemini_api_key)
    qa_context["ready"] = True
    qa_context["initializing"] = False
    qa_context["error"] = None


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():
    user_message = request.form.get("msg", "").strip()

    if not user_message:
        return "Please enter a medical question."

    if not qa_context["ready"]:
        try:
            initialize_chatbot()
        except Exception as exc:
            qa_context["ready"] = False
            qa_context["initializing"] = False
            qa_context["error"] = str(exc)
            return (
                "The medical chatbot backend could not initialize. "
                f"Error: {qa_context['error']}"
            )

    try:
        from src.helper import answer_with_gemini, get_relevant_docs

        docs = get_relevant_docs(qa_context["docsearch"], user_message, top_k=3)
        answer = answer_with_gemini(
            qa_context["gemini_client"],
            user_message,
            docs,
        )
        return answer or "I do not know based on the retrieved medical information."
    except Exception as exc:
        return f"Sorry, I could not generate an answer right now. Error: {exc}"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False, use_reloader=False)
