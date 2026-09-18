import json
import logging
import time
from pathlib import Path


def get_logger(name: str, log_dir: str = "logs") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh = logging.FileHandler(Path(log_dir) / f"{name}.log")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def log_metrics(
    logger: logging.Logger,
    metrics: dict,
    step: str = "",
    json_path: str | None = None,
) -> None:
    msg = f"[{step}] " if step else ""
    msg += " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
    logger.info(msg)
    if json_path:
        Path(json_path).parent.mkdir(parents=True, exist_ok=True)
        payload = {"step": step, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **metrics}
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)
