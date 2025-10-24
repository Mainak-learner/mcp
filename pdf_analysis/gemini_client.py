# gemini_client.py
import os, sys, json, argparse
import google.generativeai as genai
import retrieval_core as core  # reuse the same retrieval locally

PROMPT = """You are a precise assistant. Use ONLY the provided snippets for each document.
For EACH document, write concise bullet points answering the user's question.
If a document lacks relevant info, say so for that document. Avoid hallucination.

Return Markdown ONLY in this structure:
## <filename-1>
- bullet
- bullet

## <filename-2>
- bullet
- bullet

Overall: <one-line cross-document synthesis if useful>

Context (grouped by document):
-----
{context_md}

Question: {question}
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--paths", nargs="*", default=None, help="Optional PDFs/dirs to (re)index now")
    ap.add_argument("--model", default=os.environ.get("GEMINI_MODEL","gemini-1.5-pro"))
    ap.add_argument("--top_k", type=int, default=24)
    ap.add_argument("--max_ctx_chars", type=int, default=9000)
    args = ap.parse_args()

    # Optional re-index now
    if args.paths:
        print(core.index_pdfs(args.paths)["message"])

    # Build context locally (no data leaves your machine until calling Gemini)
    ctx = core.retrieve_context(args.question, top_k=args.top_k, files=None, max_ctx_chars=args.max_ctx_chars)
    if not ctx.get("ok"):
        print(ctx.get("message","retrieval failed"), file=sys.stderr); sys.exit(1)
    if not ctx["context_md"].strip():
        print("No context under budget. Increase --max_ctx_chars.", file=sys.stderr); sys.exit(1)

    api_key = os.environ.get("AIzaSyCTnv7TT6uQB_IYrJOeJkruIkHEFjuiK7A")
    if not api_key:
        raise RuntimeError("Set GOOGLE_API_KEY")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    prompt = PROMPT.format(context_md=ctx["context_md"], question=args.question)
    resp = model.generate_content(prompt)
    print(resp.text)

if __name__ == "__main__":
    main()
