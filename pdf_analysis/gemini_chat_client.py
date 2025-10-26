# gemini_chat_client.py (quiet chat)
import os, sys, json, asyncio
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters

# Gemini SDK (new)
from google import genai
from google.genai import types

# ---- Silence noisy libs BEFORE anything loads TF/hf/tqdm ----
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")          # hide TF INFO/WARN
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")          # kills that oneDNN note
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = (
    "You are a careful assistant with access to PDF tools via MCP. "
    "Keep answers concise. When I ask about PDFs, call the tools as needed, "
    "then answer based ONLY on the retrieved snippets."
)

def build_gemini_tools_from_mcp(tools_resp):
    decls = []
    want = {"index_pdfs", "retrieve_context", "list_indexed_files", "clear_index"}
    for t in tools_resp.tools:
        if t.name not in want:
            continue
        # Minimal schema passthrough – all args go under {"input": ...}
        decls.append(types.FunctionDeclaration(
            name=t.name,
            description=t.description or "",
            parameters=types.Schema(type=types.Type.OBJECT)  # allow free-form
        ))
    return [types.Tool(function_declarations=decls)]

def extract_function_calls(resp):
    calls = []
    for cand in (getattr(resp, "candidates", None) or []):
        content = getattr(cand, "content", None)
        for p in (getattr(content, "parts", None) or []):
            fc = getattr(p, "function_call", None)
            if fc:
                calls.append((fc.name, dict(fc.args or {}), getattr(fc, "id", None)))
    for fc in (getattr(resp, "function_calls", None) or []):
        calls.append((fc.name, dict(fc.args or {}), getattr(fc, "id", None)))
    return calls

def make_function_response_part(name, response_obj, id_like=None):
    return types.Part(function_response=types.FunctionResponse(name=name, response=response_obj))

def get_text(resp) -> str:
    """Collect only text parts (avoids resp.text and its warnings)."""
    out = []
    for cand in (getattr(resp, "candidates", None) or []):
        content = getattr(cand, "content", None)
        for p in (getattr(content, "parts", None) or []):
            if getattr(p, "text", None):
                out.append(p.text)
    return "\n".join(out).strip()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python gemini_chat_client.py /abs/path/to/pdf_qa_server.py")
        return

    server_script = sys.argv[1]

    # Gemini client
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY")
    ai = genai.Client(api_key=api_key)

    # Launch MCP server with quiet env
    params = StdioServerParameters(
        command="python",
        args=[server_script],
        env={
            "TF_CPP_MIN_LOG_LEVEL": "3",
            "TF_ENABLE_ONEDNN_OPTS": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "TQDM_DISABLE": "1",
            "TRANSFORMERS_VERBOSITY": "error",
        },
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_resp = await session.list_tools()
            gemini_tools = build_gemini_tools_from_mcp(tools_resp)

            config = types.GenerateContentConfig(
                tools=gemini_tools,
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=800,
            )
            contents: list[types.Content] = []

            while True:
                try:
                    user_text = input("You: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nBye!")
                    return
                if not user_text:
                    continue
                if user_text.lower() in {"exit", "quit", "q"}:
                    print("Bye!")
                    return

                contents.append(types.Content(role="user", parts=[types.Part(text=user_text)]))
                resp = ai.models.generate_content(model=MODEL, contents=contents, config=config)

                while True:
                    # Print assistant message (text only)
                    msg = get_text(resp)
                    if msg:
                        print("\nAssistant:", msg)

                    fn_calls = extract_function_calls(resp)
                    if not fn_calls:
                        if resp.candidates and resp.candidates[0].content:
                            contents.append(resp.candidates[0].content)
                        break  # back to top-level input

                    # Execute exactly one tool call (quietly)
                    name, args, fc_id = fn_calls[0]
                    # Wrap arguments under {"input": ...} for FastMCP tools
                    mcp_args = {"input": args} if name in {
                        "index_pdfs", "retrieve_context", "list_indexed_files", "clear_index"
                    } else args
                    tool_result = await session.call_tool(name, mcp_args)

                    # Normalize result payload for function_response
                    result_payload = []
                    for part in tool_result.content:
                        if part.type == "json":
                            result_payload.append(part.data)
                        elif part.type == "text":
                            result_payload.append({"text": part.text})

                    # Return tool results to model
                    function_response_content = types.Content(
                        role="user",
                        parts=[make_function_response_part(name, {"content": result_payload}, id_like=fc_id)]
                    )
                    convo_plus = contents + [resp.candidates[0].content, function_response_content]
                    resp = ai.models.generate_content(model=MODEL, contents=convo_plus, config=config)

                    # Keep the convo history tight but complete
                    contents = convo_plus + ([resp.candidates[0].content] if resp.candidates else [])

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBye!")
