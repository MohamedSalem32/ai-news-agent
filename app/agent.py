# app/agent.py

from __future__ import annotations
import os

def build_react_agent(data_dir: str, use_st: bool | str = "auto"):
    from .tools import build_tools
    tools_pkg = build_tools(data_dir, use_st=use_st)
    lc_tools = tools_pkg.get("_lc_tools", None)

    class InteractiveFallbackCLI:
        def run(self):
            print("Fast Reading Agent (fallback CLI). Type 'help' for commands, 'quit' to exit.")
            while True:
                try:
                    cmd = input("> ").strip()
                except EOFError:
                    break
                if cmd in ("quit", "exit"):
                    break
                if cmd == "help":
                    print("focus <topic> <top_n=3>")
                    print("digest <topic> <n=5>")
                    continue
                if cmd.startswith("focus"):
                    parts = cmd.split(maxsplit=2)
                    if len(parts) >= 2:
                        topic = parts[1] if len(parts) == 2 else parts[1] + " " + parts[2]
                        top_n = 3
                        if parts[-1].isdigit():
                            top_n = int(parts[-1]); topic = " ".join(parts[1:-1]) or topic
                        res = tools_pkg["focus"](topic, top_n)
                        from pprint import pprint; pprint(res)
                    else:
                        print("Usage: focus <topic> <top_n=3>")
                elif cmd.startswith("digest"):
                    parts = cmd.split(maxsplit=2)
                    if len(parts) >= 2:
                        topic = parts[1] if len(parts) == 2 else parts[1] + " " + parts[2]
                        n = 5
                        if parts[-1].isdigit():
                            n = int(parts[-1]); topic = " ".join(parts[1:-1]) or topic
                        res = tools_pkg["digest"](topic, n)
                        from pprint import pprint; pprint(res)
                    else:
                        print("Usage: digest <topic> <n=5>")
                else:
                    print("Unknown command. Type 'help'.")

    if lc_tools is None:
        return InteractiveFallbackCLI()

    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain_community.chat_models import ChatOllama
    except Exception as e:
        print(f"[warn] LangChain imports failed: {e}\nUsing fallback CLI.")
        return InteractiveFallbackCLI()

    # STRICT ReAct prompt: include {tools}, {tool_names}, {input}, {agent_scratchpad}
    prompt = ChatPromptTemplate.from_template(
    "You are a concise news assistant.\n\n"
    "You have access to these tools:\n{tools}\n\n"
    "ROUTING RULES\n"
    "- If the user says 'focus on <topic> [N]', call focus with top_n=N (default 5).\n"
    "- If the user says 'digest <topic> [N]', call digest with n=N (default 5).\n"
    "- Otherwise, prefer focus on the user's text as <topic>.\n\n"
    "RESPONSE FORMAT (always):\n"
    "Final Answer:\n"
    "**Top {N} for \"<topic>\"**\n"
    "{% for i,item in enumerate(results, start=1) %}"
    "{i}. **{item.title}** — {item.source} ({item.date or 'n/a'})\n"
    "   - Why ranked: relevance={item.relevance:.2f}, recency={item.recency:.2e}, cred={item.credibility:.2f}, score={item.score:.2f}\n"
    "   - Link: {item.url}\n"
    "{% endfor %}\n\n"
    "Example tool call:\n"
    "Thought: I should retrieve ranked articles.\n"
    "Action: focus\n"
    "Action Input: {{\"topic\": \"AI in healthcare\", \"top_n\": 3}}\n"
    "Observation: <tool returns JSON>\n"
    "Thought: I will format the results per the RESPONSE FORMAT.\n"
    "Final Answer: <as specified above>\n\n"
    "Question: {input}\n"
    "{agent_scratchpad}"
    )



    base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    # llm = ChatOllama(model="mistral", temperature=0, base_url=base_url)
    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "mistral:7b-instruct-q2_K"),
        base_url=base_url,
        temperature=0,
       model_kwargs={"num_gpu": 0, "num_ctx": 1024}
                        )
     


    agent = create_react_agent(llm, lc_tools, prompt)

    executor = AgentExecutor(
    agent=agent,
    tools=lc_tools,
    verbose=True,
    handle_parsing_errors=True,   # keep this True
    max_iterations=2,             # keep it tight so it doesn't loop
    early_stopping_method="force",# <-- this fixes your crash
    return_intermediate_steps=False,
    )
    return executor

 
if __name__ == "__main__":
    import re, json
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    # Build the ReAct agent (LangChain) as before
    agent = build_react_agent(data_dir, use_st="auto")
    print("Agent ready.")

    # Build tools once for the direct-bypass commands (yes, this builds a retriever too; OK for now)
    from .tools import build_tools
    tools_pkg = build_tools(data_dir, use_st="auto")

    if hasattr(agent, "run") and not hasattr(agent, "invoke"):
        agent.run()
    else:
        print("LangChain agent ready. Type a question, or 'quit' to exit.")
        while True:
            q = input("> ").strip()
            if q in ("quit", "exit"):
                break

            # --- DIRECT BYPASS: exact, deterministic (no LLM) ---
            m = re.match(r'^\s*focus\s+(.+?)\s+(\d+)\s*$', q, re.I)
            if m:
                topic, n = m.group(1), int(m.group(2))
                res = tools_pkg["focus"](topic, n)
                print(json.dumps(res, ensure_ascii=False, indent=2))
                continue

            m = re.match(r'^\s*digest\s+(.+?)\s+(\d+)\s*$', q, re.I)
            if m:
                topic, n = m.group(1), int(m.group(2))
                res = tools_pkg["digest"](topic, n)
                print(json.dumps(res, ensure_ascii=False, indent=2))
                continue

            # --- Otherwise let the ReAct agent handle natural language ---
            try:
                out = agent.invoke({"input": q})
            except Exception as e:
                print(f"[error] {e}")
                continue
            print(out.get("output", out))
