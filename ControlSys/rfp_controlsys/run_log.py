"""
The log of a run (CS-80, SY-23): one folder per run in ControlSys/runs/ (git-ignored) with
one CSV row per cycle, the annotated observation image of each cycle, and a run report.
"""

import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"
COLUMNS = (
    "run_id", "cycle", "time", "duration_s", "observation_pose", "fish_id", "decision_source",
    "decision_reason", "candidates", "grasp_x_mm", "grasp_y_mm", "grasp_z_mm", "heading_deg",
    "heading_known", "length_mm", "thickness_mm", "confidence", "slot", "attempt", "result",
    "failure_code", "failed_step", "detail",
)  # fmt: skip


class RunLog:
    def __init__(self, info: dict, root: Path | None = RUNS_DIR):
        """`info`: mode and versions, written into the report. `root=None`: nothing is written."""
        self.run_id = time.strftime("%Y-%m-%d_%H%M%S")
        self.info = dict(info, run_id=self.run_id, started=time.strftime("%Y-%m-%d %H:%M:%S"))
        self.folder = None if root is None else root / f"{self.run_id}_{info.get('mode', 'run')}"
        self._file = self._writer = None
        if self.folder is not None:
            self.folder.mkdir(parents=True, exist_ok=True)
            # stays open for the whole run: close() ends it
            self._file = open(  # noqa: SIM115
                self.folder / "cycles.csv", "w", newline="", encoding="utf-8"
            )
            self._writer = csv.DictWriter(self._file, COLUMNS, extrasaction="ignore")
            self._writer.writeheader()

    def cycle(self, row: dict) -> None:
        if self._writer is not None:
            self._writer.writerow(dict(row, run_id=self.run_id))
            self._file.flush()  # the rows survive a crash or a kill

    def image(self, name: str, image: np.ndarray) -> None:
        if self.folder is not None:
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok:
                encoded.tofile(str(self.folder / f"{name}.jpg"))  # works with non-ASCII paths

    def close(self, report: dict) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            summary = dict(self.info, ended=time.strftime("%Y-%m-%d %H:%M:%S"), **report)
            (self.folder / "report.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
