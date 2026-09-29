from app import config

REFUSAL = "I couldn't find this in the course material."

SYSTEM = f"""You are a course tutor. Answer ONLY using the numbered context passages.
Cite every claim inline as [source, p.X] using the passage metadata.
If the context does not contain the answer, reply exactly: "{REFUSAL}"
Do not use outside knowledge."""


def build_prompt(question: str, passages: list[dict]) -> str:
    ctx = "\n\n".join(f"[{p['source']}, p.{p['page']}]\n{p['text']}" for p in passages)
    return f"Context:\n{ctx}\n\nQuestion: {question}"


import time

def call_llm(system: str, user: str) -> tuple[str, dict]:
    if config.LLM_PROVIDER == "groq":
        from groq import Groq
        client = Groq()  # reads GROQ_API_KEY
        for attempt in range(3):
            try:
                r = client.chat.completions.create(
                    model=config.LLM_MODEL, temperature=0, max_tokens=800,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}])
                return r.choices[0].message.content, {"in": r.usage.prompt_tokens,
                                                      "out": r.usage.completion_tokens}
            except Exception as e:
                if "429" in str(e) and attempt < 2:   # free-tier rate limit
                    time.sleep(5 * (attempt + 1))
                    continue
                raise
    if config.LLM_PROVIDER == "anthropic":
        import anthropic
        r = anthropic.Anthropic().messages.create(
            model=config.LLM_MODEL, max_tokens=800, system=system,
            messages=[{"role": "user", "content": user}])
        return r.content[0].text, {"in": r.usage.input_tokens, "out": r.usage.output_tokens}
    from openai import OpenAI
    r = OpenAI().chat.completions.create(
        model=config.LLM_MODEL, temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
    return r.choices[0].message.content, {"in": r.usage.prompt_tokens, "out": r.usage.completion_tokens}

def answer(question: str, passages: list[dict]) -> tuple[str, dict]:
    if not passages or (passages[0]["score"] is not None and passages[0]["score"] < config.REFUSE_THRESHOLD):
        return REFUSAL, {"in": 0, "out": 0}
    return call_llm(SYSTEM, build_prompt(question, passages))
