from typing import List
from agent.base import MemoryAgent


class PlanningAgent(MemoryAgent):
    """分解任务并制定计划的Agent"""

    system_prompt = "你是一个擅长进行任务拆解和规划的助手,"

    def __init__(self, name, desc, llm, max_message):
        super().__init__(name, desc, llm, max_message)
        self.plans: List[str] = []


    def think(self) -> str:
        