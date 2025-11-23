# ui/dashboard/monitor.py
# Simple console-based monitor:

class Dashboard:
    def display_cycle(self, result: dict):
        print("--- Cycle Summary ---")
        for k, v in result.items():
            print(k, ":", v)