from importlib import import_module
from pathlib import Path
from typing import Dict, Optional


class ExperimentLogger:
    def __init__(
        self,
        backend: str,
        project: str,
        run_name: str,
        save_dir: Path,
        config: Dict[str, object],
    ) -> None:
        self.backend = backend
        self.project = project
        self.run_name = run_name
        self.save_dir = save_dir
        self.config = config
        self.client = None

        if backend == "none":
            return

        if backend == "wandb":
            wandb = import_module("wandb")
            self.client = wandb
            wandb.init(
                project=project,
                name=run_name,
                config=config,
                dir=str(save_dir),
            )
            return

        if backend == "swanlab":
            swanlab = import_module("swanlab")
            self.client = swanlab
            swanlab.init(
                project=project,
                experiment_name=run_name,
                config=config,
                logdir=str(save_dir),
            )
            return

        raise ValueError(f"Unsupported logger backend: {backend}")

    def log(self, metrics: Dict[str, object], step: Optional[int] = None) -> None:
        if self.backend == "none" or self.client is None:
            return

        if self.backend == "wandb":
            self.client.log(metrics, step=step)
            return

        if self.backend == "swanlab":
            if step is not None:
                metrics = {"step": step, **metrics}
            self.client.log(metrics)

    def finish(self) -> None:
        if self.backend == "none" or self.client is None:
            return

        if self.backend == "wandb":
            self.client.finish()
            return

        if self.backend == "swanlab":
            finish_fn = getattr(self.client, "finish", None)
            if callable(finish_fn):
                finish_fn()
