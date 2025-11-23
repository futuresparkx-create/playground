# main.py

from orchestrator.graph import ImprovementGraph
from ui.dashboard.monitor import Dashboard
import yaml

config = {
    "model": yaml.safe_load(open("config/model.yaml")),
    "cycles": yaml.safe_load(open("config/cycles.yaml"))["cycles"]
}

graph = ImprovementGraph(config)
dashboard = Dashboard()

if __name__ == "__main__":
    task = input("Enter coding task: ")
    result = graph.cycle(task)
    dashboard.display_cycle(result)

    print("\nReview the output manually before continuing.")