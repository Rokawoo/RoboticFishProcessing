"""
The operator's side of a run (CS-70): states and messages on the terminal, the camera
image in a window, a stop key, and alerts that wait for the operator (CS-74). The cycle
only talks to these few methods, so another interface can replace this one.
"""

import sys

import cv2
import numpy as np
from rfp_vision.overlay import add_status

WINDOW = "Pick and place (q or Esc: stop)"
WINDOW_MAX = (1280, 800)  # the window fits in this (px)


def ask(prompt: str) -> str | None:
    """The operator's answer, lower case; None if nobody can answer (input closed)."""
    try:
        return input(prompt).strip().lower()
    except EOFError:
        print()
        return None


class Console:
    def __init__(self, window: bool = True, interactive: bool | None = None, step: bool = False):
        """
        `window`: show the camera image (its q/Esc key and closing it stop the run).
        `interactive`: alerts wait for Enter; default: only if a person is at the terminal.
        `step`: every move waits for Enter before it runs.
        """
        self.window = window
        self.interactive = sys.stdin.isatty() if interactive is None else interactive
        self.step = step
        self._state = "idle"
        self._detail = ""
        self._stop = False
        self._shown = False

    # -- terminal --------------------------------------------------------------
    def state(self, state: str, detail: str = "") -> None:
        """A system state of CS-72, printed when it changes."""
        if (state, detail) != (self._state, self._detail):
            self._state, self._detail = state, detail
            print(f"[{state}] {detail}".rstrip())

    def info(self, text: str) -> None:
        print(f"   {text}")

    def alert(self, message: str) -> bool:
        """An alert that stays until the operator answers. True: resume; False: stop the run."""
        print(f"\n⚠️  {message}")
        if not self.interactive:
            print("   Nobody at the terminal to acknowledge it: stopping the run.")
            return False
        answer = ask("   Enter: resume    q + Enter: stop the run  > ")
        return answer is not None and answer not in ("q", "quit", "stop")

    def confirm_move(self, description: str) -> bool:
        """Step mode: ask before a planned move runs. False cancels it and stops the run."""
        if not self.step:
            return True
        if not self.interactive:
            return False
        answer = ask(f"   next: {description}\n   Enter: move    q + Enter: stop  > ")
        return answer is not None and answer not in ("q", "quit", "stop")

    # -- window ----------------------------------------------------------------
    def show(self, image: np.ndarray, lines: list[str] | None = None) -> None:
        """Show a camera image under the state line and `lines`; reads the stop key."""
        if not self.window:
            return
        image = add_status(image, [f"{self._state}  {self._detail}".rstrip(), *(lines or [])])
        h, w = image.shape[:2]
        fit = min(WINDOW_MAX[0] / w, WINDOW_MAX[1] / h, 1.0)
        if fit < 1:
            image = cv2.resize(
                image, (round(w * fit), round(h * fit)), interpolation=cv2.INTER_AREA
            )
        try:
            cv2.imshow(WINDOW, image)
            key = cv2.waitKey(1) & 0xFF
            closed = self._shown and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:  # no display: go on without the window
            self.window = False
            print("   (no display for the camera window: continuing without it)")
            return
        self._shown = True
        if key in (ord("q"), 27) or closed:
            self._stop = True

    def stop_requested(self) -> bool:
        return self._stop

    def close(self) -> None:
        if self._shown:
            cv2.destroyAllWindows()
