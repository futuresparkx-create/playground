#orchestrator/graph.py
# ORCHESTRATOR (LangGraph-Compatible Skeleton)

from orchestrator.nodes.generate import GenerateNode
from orchestrator.nodes.reflect import ReflectNode
from orchestrator.nodes.test import TestNode
from orchestrator.nodes.learn import LearnNode

class ImprovementGraph:
    """
    Safe, deterministic graph. No autonomous recursion.
    Human must invoke cycle() manually.
    """

    def __init__(self, config):
        self.config = config
        self.generate_node = GenerateNode()
        self.reflect_node = ReflectNode()
        self.test_node = TestNode()
        self.learn_node = LearnNode()

        self.cycles_run = 0
        self.reflects_run = 0

    def cycle(self, input_task: str):
        """
        Run one safe cycle of the improvement loop.
        """

        if self.cycles_run >= self.config["cycles"]["max_cycles"]:
            raise RuntimeError("Cycle limit reached. Human review required.")

        self.cycles_run += 1

        # Step 1: Generate
        gen = self.generate_node.run(input_task)

        # Step 2: Reflect (optional and bounded)
        if self.reflects_run < self.config["cycles"]["max_reflect"]:
            self.reflects_run += 1
            reflection = self.reflect_node.run(gen["output"])
        else:
            reflection = None

        # Step 3: Test (safe: static only by default)
        test = self.test_node.run(gen["output"])

        # Step 4: Learn (logs only, no weight changes)
        learn = self.learn_node.run({
            "generation": gen,
            "reflection": reflection,
            "test": test
        })

        return {
            "generation": gen,
            "reflection": reflection,
            "test": test,
            "learn": learn
        }
