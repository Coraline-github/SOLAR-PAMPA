# Base Agent class for SOLAR-PAMPA pipeline
# All agents inherit from BaseAgent
# Every agent must implement: validate_inputs() and run()

from abc import ABC, abstractmethod
from pathlib import Path
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s"
)

class BaseAgent(ABC):

    def __init__(self, name: str, input_path: Path, output_path: Path):
        self.name        = name
        self.input_path  = input_path
        self.output_path = output_path
        self.logger      = logging.getLogger(name)
        self.status      = "idle"

        # Always make sure output folder exists
        self.output_path.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def validate_inputs(self) -> bool:
        """Verify required input files exist before running."""
        pass

    @abstractmethod
    def run(self) -> bool:
        """Core agent logic. Must return True on success."""
        pass

    def execute(self) -> bool:
        """
        Called by pipeline.py — wraps run() with:
        - timing
        - logging
        - error handling
        """
        self.logger.info(f"Starting {self.name}...")
        self.status = "running"
        start_time  = time.time()

        try:
            if not self.validate_inputs():
                raise ValueError(f"{self.name}: Input validation failed")

            success     = self.run()
            elapsed     = round(time.time() - start_time, 2)
            self.status = "done" if success else "failed"

            self.logger.info(f"{self.name} finished in {elapsed}s — {self.status}")
            return success

        except Exception as e:
            self.status = "failed"
            self.logger.error(f"{self.name} crashed: {e}")
            raise