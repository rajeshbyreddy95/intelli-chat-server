from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
from dotenv import load_dotenv
from langchain_perplexity import ChatPerplexity
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
import traceback
import os

load_dotenv()
app = Flask(__name__)
CORS(app)

# Initialize LLM
llm = ChatPerplexity(
    model="sonar",
    temperature=0.6,
)

# Store conversations in memory (dict: user_id -> list of messages)
conversations = {}

prompt_template = """
You are a professional summarizer.
Summarize the following page content into a short paragraph.
Add an HTML <br> after each sentence.

Page content:
{page_text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["page_text"])


@app.route('/summarize', methods=['POST'])
def summarize():
    page_data = {}
    file = request.files.get('file')

    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    try:
        with pdfplumber.open(file) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text or not text.strip():
                    page_data[f"Page {i+1}"] = "No readable text on this page.<hr>"
                    continue

                final_prompt = prompt.format(page_text=text)
                response = llm.invoke([HumanMessage(content=final_prompt)])
                page_data[f"Page {i+1}"] = response.content.strip() + "<hr>"

        return jsonify({"summary": page_data})
    except Exception as e:
        print("Error:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_input = (data.get("message") or "").strip()
        user_id = data.get("user_id", "default")  # frontend should pass user_id
        print(f"[{user_id}] {user_input}")

        if not user_input:
            return jsonify({"error": "Message is required"}), 400

        # Initialize conversation if new user
        if user_id not in conversations:
            conversations[user_id] = []

        # Append user message
        conversations[user_id].append({"role": "user", "content": user_input})

        # Build conversation history for LLM
        history = [HumanMessage(content=msg["content"]) if msg["role"] == "user" 
                   else HumanMessage(content=msg["content"])  # adjust if system/assistant roles needed
                   for msg in conversations[user_id]]

        response = llm.invoke(history)

        # Append assistant response to conversation
        conversations[user_id].append({"role": "assistant", "content": response.content.strip()})

        return jsonify({"response": response.content.strip(), "history": conversations[user_id]})

    except Exception as e:
        print("Error in /chat:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/ask_with_context', methods=['POST'])
def ask_with_context():
    """
    Accepts JSON: { "message": "...", "page_text": "..." }
    Uses the page_text as context, then asks the llm to answer.
    """
    try:
        data = request.get_json()
        user_input = (data.get("message") or "").strip()
        page_text = (data.get("page_text") or "").strip()

        if not user_input:
            return jsonify({"error": "Message is required"}), 400

        system_prompt = (
            "You are an assistant that must answer user questions using the provided page text. "
            "If the answer is present in the page_text, answer concisely and cite that it came from the page. "
            "If page_text doesn't contain the answer, say you don't know and suggest next steps."
        )

        combined_prompt = (
            f"{system_prompt}\n\nPage content:\n{page_text}\n\nUser question:\n{user_input}"
        )

        response = llm.invoke([HumanMessage(content=combined_prompt)])
        return jsonify({"response": response.content.strip()})

    except Exception as e:
        print("Error in /ask_with_context:", e)
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5040, debug=True)
