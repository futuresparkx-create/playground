# monetization/pr_agent/agent.py

# PR Agent Scaffold (NO automation, safe-only)*

class PRAgent:
    """
    Safe: generates patch suggestions only.
    Does not submit PRs, run code, or modify repos.
    """

    def propose_patch(self, diff: str):
        return "[PATCH_SUGGESTION_PLACEHOLDER]"

    def explain_patch(self, patch: str):
        return "[PATCH_EXPLANATION_PLACEHOLDER]"