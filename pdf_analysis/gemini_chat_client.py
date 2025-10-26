# client-gemini.py (interactive; PDF RAG only)
import os, sys, json, asyncio
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

# Gemini SDK (pip install -U google-genai)
from google import genai
from google.genai import types
import warnings

warnings.filterwarnings(
    "ignore",
    message="there are non-text parts in the response",
    category=UserWarning,
)

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = (
    "You are a careful assistant with access to PDF tools via MCP.\n"
    "\n"
    "PDF RULES:\n"
    "1) For any question about the PDFs, first call `retrieve_context` with the full user question.\n"
    "2) The tool returns JSON containing `context_md` (snippets grouped by filename). Use ONLY that context.\n"
    "3) Answer in EXACTLY this Markdown structure:\n"
    "   ## <filename-1>\n"
    "   - bullet\n"
    "   - bullet\n"
    "\n"
    "   ## <filename-2>\n"
    "   - bullet\n"
    "   - bullet\n"
    "\n"
    "   Overall: <one-line cross-document synthesis if useful>\n"
    "4) If the user asks to (re)index PDFs, call `index_pdfs` with the provided paths (files or folders).\n"
    "   - After indexing, call `list_indexed_files` and show the basenames.\n"
    "5) Use `list_indexed_files` to show what's indexed. Use `clear_index` only if the user asks to reset.\n"
    "6) Do not answer from memory for PDF questions; always retrieve first.\n"
)

def build_gemini_tools_from_mcp(tools_resp):
    """Convert MCP tool metadata → Gemini function declarations (PDF tools only)."""
    decls = []
    for t in tools_resp.tools:
        if t.name == "index_pdfs":
            params = types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "paths": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.STRING),
                        description="PDF files or directories",
                    ),
                    "chunk_chars": types.Schema(type=types.Type.INTEGER),
                    "overlap": types.Schema(type=types.Type.INTEGER),
                },
                required=["paths"],
            )
            decls.append(
                types.FunctionDeclaration(
                    name="index_pdfs",
                    description=t.description or "Index one or more PDFs (or folders).",
                    parameters=params,
                )
            )

        elif t.name == "retrieve_context":
            params = types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "question": types.Schema(type=types.Type.STRING, description="User question"),
                    "top_k": types.Schema(type=types.Type.INTEGER),
                    "files": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
                    "max_ctx_chars": types.Schema(type=types.Type.INTEGER),
                },
                required=["question"],
            )
            decls.append(
                types.FunctionDeclaration(
                    name="retrieve_context",
                    description=t.description or "Retrieve per-document snippets and `context_md`.",
                    parameters=params,
                )
            )

        elif t.name == "list_indexed_files":
            decls.append(
                types.FunctionDeclaration(
                    name="list_indexed_files",
                    description=t.description or "Show indexed PDF basenames.",
                    parameters=types.Schema(type=types.Type.OBJECT, properties={}),
                )
            )

        elif t.name == "clear_index":
            decls.append(
                types.FunctionDeclaration(
                    name="clear_index",
                    description=t.description or "Delete FAISS index + metadata.",
                    parameters=types.Schema(type=types.Type.OBJECT, properties={}),
                )
            )

    return [types.Tool(function_declarations=decls)]

def extract_function_calls(resp):
    """Return [(name, args, id_like), ...] from a Gemini response."""
    calls = []
    for cand in (getattr(resp, "candidates", None) or []):
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for p in parts:
            fc = getattr(p, "function_call", None)
            if fc:
                calls.append((fc.name, dict(fc.args or {}), getattr(fc, "id", None)))
    for fc in (getattr(resp, "function_calls", None) or []):
        calls.append((fc.name, dict(fc.args or {}), getattr(fc, "id", None)))
    return calls

def make_function_response_part(name, response_obj, id_like=None):
    fr = types.FunctionResponse(name=name, response=response_obj)
    return types.Part(function_response=fr)

def _is_exit(s: str) -> bool:
    return s.strip().lower() in {"exit", "quit", "q"}

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client-gemini.py /absolute/path/to/pdf_qa_server.py")
        return

    server_script = sys.argv[1]
    params = StdioServerParameters(command="python", args=[server_script])

    # Gemini client (reads GOOGLE_API_KEY from env)
    api_key = os.getenv("AIzaSyCTnv7TT6uQB_IYrJOeJkruIkHEFjuiK7A")
    if not api_key:
        raise ValueError(
            "Set GOOGLE_API_KEY in your environment (or pass api_key to genai.Client)."
        )
    ai = genai.Client(api_key=api_key)


    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Discover MCP tools + advertise to Gemini
            tools_resp = await session.list_tools()
            gemini_tools = build_gemini_tools_from_mcp(tools_resp)

            config = types.GenerateContentConfig(
                tools=gemini_tools,
                system_instruction=SYSTEM_PROMPT,
                temperature=0.1,
                max_output_tokens=1200,
            )
            contents: list[types.Content] = []

            print("PDF assistant ready. Ask things like:")
            print("- index /content/pdfs and /content/extra")
            print("- list indexed files")
            print("- who are the authors of each paper?")
            print("- clear the index")
            print("Type 'exit' to quit.\n")

            while True:
                user_text = input("You: ").strip()
                if _is_exit(user_text):
                    print("Bye!")
                    return
                if not user_text:
                    continue

                contents.append(types.Content(role="user", parts=[types.Part(text=user_text)]))
                resp = ai.models.generate_content(model=MODEL, contents=contents, config=config)

                while True:
                    progressed = False

                    if getattr(resp, "text", None):
                        print("\nAssistant:", resp.text)

                    fn_calls = extract_function_calls(resp)
                    if not fn_calls:
                        # record assistant turn
                        if resp.candidates and resp.candidates[0].content:
                            contents.append(resp.candidates[0].content)
                        break  # back to top for next user turn

                    # Handle tool calls one by one
                    for (name, args, fc_id) in fn_calls:
                        print(f"\n[Tool use requested] {name} with args {json.dumps(args, indent=2)}")

                        # Call MCP tool
                        # NOTE: your pdf_qa_server tools accept top-level args (no {"input":...} wrapper)
                        tool_result = await session.call_tool(name, args)

                        # Normalize MCP result parts to plain JSON for Gemini
                        result_payload = []
                        for part in tool_result.content:
                            if part.type == "json":
                                result_payload.append(part.data)
                            elif part.type == "text":
                                # server returns JSON strings sometimes — try to parse, else wrap
                                try:
                                    result_payload.append(json.loads(part.text))
                                except Exception:
                                    result_payload.append({"text": part.text})
                            else:
                                result_payload.append({"type": part.type})

                        # Send function_response back to the model, let it finalize
                        function_response_content = types.Content(
                            role="user",
                            parts=[make_function_response_part(name, {"content": result_payload}, id_like=fc_id)]
                        )
                        convo_plus = contents + [resp.candidates[0].content, function_response_content]
                        resp = ai.models.generate_content(model=MODEL, contents=convo_plus, config=config)

                        if getattr(resp, "text", None):
                            print("\nAssistant:", resp.text)

                        contents = convo_plus + ([resp.candidates[0].content] if resp.candidates else [])

                        next_calls = extract_function_calls(resp)
                        if not next_calls:
                            progressed = True
                            break

                    if progressed:
                        break

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBye!")
