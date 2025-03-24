from app.llm import LLM


class BaseAgent:
    name: str
    desc: str

    llm: LLM
    state: int
    
