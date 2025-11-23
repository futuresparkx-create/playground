# orchestrator/nodes/learn.py
# Learn Node (Real LanceDB Integration)

from memory.memory import MemoryStore

class LearnNode:
    """
    Logs only. No training. No model updates.
    """

    def __init__(self):
        self.memory = MemoryStore()

    def run(self, data: dict):
        # Log the full episode
        self.memory.log_episode(data)

        # Store semantic content
        if "generation" in data:
            emb = self.memory.embed(data["generation"]["output"])
            self.memory.store_semantic(emb, data["generation"])

        return {"logged": True}