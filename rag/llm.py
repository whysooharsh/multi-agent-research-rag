import os

from rag import config


def get_llm(temperature: float = 0.0):
    if os.getenv("LLM_PROVIDER", "gemini") == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=temperature,
            base_url=os.getenv("OPENAI_BASE_URL"),
            timeout=60,
            max_retries=4,
        )
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=config.LLM_MODEL, temperature=temperature, timeout=60, max_retries=4
    )


def text_of(msg) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(p if isinstance(p, str) else p.get("text", "") for p in c)
