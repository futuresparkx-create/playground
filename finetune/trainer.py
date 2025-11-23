# finetune/trainer.py
# Fine-Tuning Pipeline (Human-Gated Skeleton)

class FineTunePipeline:
    """
    Safe: prepares datasets, does not run training automatically.
    Human must call run_training().
    """

    def __init__(self):
        pass

    def prepare_dataset(self, episodes: list):
        # TODO: format into SFT or DPO
        return "dataset_ready"

    def run_training(self, dataset_path: str):
        """
        This method must only be invoked manually by the developer.
        """
        print("Run Axolotl / ReFT training manually.")
