
import ast
import io
import re
import time
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from PIL import Image, ImageDraw

try:
    import cv2
except Exception:
    cv2 = None


Cell = Tuple[int, int]
BBox = Tuple[int, int, int, int]  # x0, y0, x1, y1


DEFAULT_ROCKS = {
    (1, 5),
    (2, 6),
    (3, 6),
    (4, 3),
    (4, 6),
    (6, 4),
    (6, 6),
    (7, 2),
    (7, 4),
    (7, 7),
}

DEFAULT_BOMBS = {
    (7, 2): 1,
    (1, 5): 2,
    (7, 7): 3,
}

DEFAULT_START = (7, 1)

DEFAULT_PATH = [
    (7, 1),
    (6, 1),
    (5, 1),
    (4, 1),
    (3, 1),
    (2, 1),
    (1, 1),
    (1, 2),
    (2, 2),
    (3, 2),
    (4, 2),
    (5, 2),
    (6, 2),
    (7, 2),
    (7, 3),
    (6, 3),
    (5, 3),
    (5, 4),
    (4, 4),
    (4, 5),
    (3, 5),
    (2, 5),
    (2, 4),
    (3, 4),
    (3, 3),
    (2, 3),
    (1, 3),
    (1, 4),
    (1, 5),
    (1, 6),
    (1, 7),
    (2, 7),
    (3, 7),
    (4, 7),
    (5, 7),
    (6, 7),
    (7, 7),
    (7, 6),
    (7, 5),
    (6, 5),
    (5, 5),
    (5, 6),
]


# ============================================================
# Text parsing
# ============================================================
def cell_to_text(cells: Set[Cell]) -> str:
    return ", ".join(f"({r},{c})" for r, c in sorted(cells))


def path_to_text(path: List[Cell]) -> str:
    return "->".join(f"({r},{c})" for r, c in path)


def parse_cell_set(text: str) -> Set[Cell]:
    pairs = re.findall(r"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?", text)
    return {(int(r), int(c)) for r, c in pairs}


def parse_cell(text: str) -> Cell:
    pairs = re.findall(r"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?", text)
    if not pairs:
        raise ValueError("No valid cell found. Use row,col, for example 7,1.")
    r, c = pairs[0]
    return int(r), int(c)


def parse_bombs(text: str) -> Dict[Cell, int]:
    if not text.strip():
        return {}
    try:
        value = ast.literal_eval(text)
        if isinstance(value, dict):
            return {(int(k[0]), int(k[1])): int(v) for k, v in value.items()}
    except Exception:
        pass

    bombs = {}
    for line in text.splitlines():
        m = re.search(r"\(?\s*(\d+)\s*,\s*(\d+)\s*\)?\s*:\s*(\d+)", line)
        if m:
            bombs[(int(m.group(1)), int(m.group(2)))] = int(m.group(3))
    return bombs


def parse_path(text: str) -> List[Cell]:
    if "->" in text:
        return [parse_cell(part) for part in text.split("->") if part.strip()]
    return list(parse_cell_set(text))


def validate_path(path: List[Cell], nrows: int, ncols: int) -> List[str]:
    warnings = []
    if len(path) != len(set(path)):
        warnings.append("The route revisits at least one cell.")
    for r, c in path:
        if not (1 <= r <= nrows and 1 <= c <= ncols):
            warnings.append(f"Cell {(r, c)} is outside the board.")
    for a, b in zip(path, path[1:]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            warnings.append(f"Non-adjacent move: {a} -> {b}.")
    return warnings


# ============================================================
# Computer vision recognition
# ============================================================
def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def find_board_bbox_auto(image: Image.Image) -> Optional[BBox]:
    """
    Heuristic board-boundary detector.

    It searches for the largest roughly square brown/yellow region, which
    corresponds to the board area in the game screenshot. This is not intended
    to be perfect. The UI includes manual correction sliders after detection.
    """
    if cv2 is None:
        return None

    arr = pil_to_rgb_array(image)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    h, w = arr.shape[:2]

    # Broad mask for board colors: brown soil + tan rock-cell background.
    # OpenCV HSV H is 0-179.
    lower = np.array([5, 20, 35])
    upper = np.array([45, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=4)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        ratio = bw / max(bh, 1)
        if area < 0.05 * w * h:
            continue
        if 0.70 <= ratio <= 1.35:
            candidates.append((area, x, y, x + bw, y + bh))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    _, x0, y0, x1, y1 = candidates[0]

    # Slight padding to include grid border.
    pad_x = int(0.005 * (x1 - x0))
    pad_y = int(0.005 * (y1 - y0))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w, x1 + pad_x)
    y1 = min(h, y1 + pad_y)

    return int(x0), int(y0), int(x1), int(y1)


def crop_cell(arr: np.ndarray, bbox: BBox, row: int, col: int, nrows: int, ncols: int, margin: float) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    cell_w = (x1 - x0) / ncols
    cell_h = (y1 - y0) / nrows

    xa = int(x0 + (col - 1 + margin) * cell_w)
    xb = int(x0 + (col - margin) * cell_w)
    ya = int(y0 + (row - 1 + margin) * cell_h)
    yb = int(y0 + (row - margin) * cell_h)

    xa, xb = max(0, xa), min(arr.shape[1], xb)
    ya, yb = max(0, ya), min(arr.shape[0], yb)

    return arr[ya:yb, xa:xb]


def detect_cells_from_bbox(
    image: Image.Image,
    bbox: BBox,
    nrows: int,
    ncols: int,
    rock_threshold: float = 0.08,
    start_threshold: float = 0.08,
    margin: float = 0.12,
) -> Tuple[Set[Cell], Optional[Cell], Dict[Cell, Dict[str, float]]]:
    """
    Cell classifier.

    Rock rule:
    Detects gray/dark low-saturation pixels inside each cell.

    Start rule:
    Detects green pixels. The cell with the highest green ratio is selected
    as the start if it exceeds the threshold.

    Returns:
    - rocks: detected rock cells
    - start: detected start cell or None
    - diagnostics: per-cell rock_ratio and green_ratio
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed. Install opencv-python.")

    arr = pil_to_rgb_array(image)
    diagnostics: Dict[Cell, Dict[str, float]] = {}
    rocks: Set[Cell] = set()
    best_green_cell: Optional[Cell] = None
    best_green_ratio = 0.0

    for r in range(1, nrows + 1):
        for c in range(1, ncols + 1):
            patch = crop_cell(arr, bbox, r, c, nrows, ncols, margin=margin)
            if patch.size == 0:
                continue

            hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
            H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

            # Gray/dark rock pixels: low saturation and middle/dark value.
            # This avoids counting the bright tan background.
            rock_mask = ((S < 75) & (V > 45) & (V < 210))

            # Add a second criterion for very dark rock outlines.
            dark_mask = ((V < 95) & (S < 140))
            rock_ratio = float(np.mean(rock_mask | dark_mask))

            # Green start cell.
            green_mask = ((H >= 35) & (H <= 90) & (S > 45) & (V > 80))
            green_ratio = float(np.mean(green_mask))

            diagnostics[(r, c)] = {
                "rock_ratio": rock_ratio,
                "green_ratio": green_ratio,
            }

            if rock_ratio >= rock_threshold:
                rocks.add((r, c))

            if green_ratio > best_green_ratio:
                best_green_ratio = green_ratio
                best_green_cell = (r, c)

    start = best_green_cell if best_green_ratio >= start_threshold else None

    # A green start cell may contain gray details from the bunny/character,
    # so remove it from rocks if needed.
    if start in rocks:
        rocks.remove(start)

    return rocks, start, diagnostics


def draw_detection_overlay(
    image: Image.Image,
    bbox: BBox,
    nrows: int,
    ncols: int,
    rocks: Set[Cell],
    start: Optional[Cell],
) -> Image.Image:
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img, "RGBA")

    x0, y0, x1, y1 = bbox
    draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0, 255), width=5)

    cell_w = (x1 - x0) / ncols
    cell_h = (y1 - y0) / nrows

    for r in range(1, nrows + 1):
        for c in range(1, ncols + 1):
            xa = x0 + (c - 1) * cell_w
            xb = x0 + c * cell_w
            ya = y0 + (r - 1) * cell_h
            yb = y0 + r * cell_h
            draw.rectangle([xa, ya, xb, yb], outline=(255, 255, 255, 160), width=1)

            if (r, c) in rocks:
                draw.rectangle([xa, ya, xb, yb], fill=(255, 165, 0, 95), outline=(255, 100, 0, 255), width=3)
                draw.text((xa + 4, ya + 4), "R", fill=(255, 0, 0, 255))

            if start == (r, c):
                draw.rectangle([xa, ya, xb, yb], fill=(0, 255, 0, 95), outline=(0, 180, 0, 255), width=3)
                draw.text((xa + 4, ya + 20), "S", fill=(0, 120, 0, 255))

    return img


# ============================================================
# Board rendering
# ============================================================
def render_board(
    nrows: int,
    ncols: int,
    rocks: Set[Cell],
    bombs: Dict[Cell, int],
    start: Cell,
    path: List[Cell],
    title: str = "Planting Path Board",
):
    fig, ax = plt.subplots(figsize=(8, 8))

    soil_color = "#8B5A2B"
    rock_bg_color = "#E8D8A8"
    start_color = "#9BE564"
    grid_edge = "#4B2E16"
    path_color = "red"

    for r in range(1, nrows + 1):
        for c in range(1, ncols + 1):
            x = c - 1
            y = r - 1

            if (r, c) == start:
                facecolor = start_color
            elif (r, c) in rocks:
                facecolor = rock_bg_color
            else:
                facecolor = soil_color

            rect = Rectangle((x, y), 1, 1, facecolor=facecolor, edgecolor=grid_edge, linewidth=1.5)
            ax.add_patch(rect)

    for (r, c) in sorted(rocks):
        cx, cy = c - 0.5, r - 0.5
        rock = Circle((cx, cy), 0.28, facecolor="gray", edgecolor="dimgray", linewidth=2)
        ax.add_patch(rock)

    path_xy = [(c - 0.5, r - 0.5) for (r, c) in path]
    for i in range(len(path_xy) - 1):
        x1, y1 = path_xy[i]
        x2, y2 = path_xy[i + 1]
        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=2.8,
            color=path_color,
            shrinkA=8,
            shrinkB=8,
            zorder=5,
        )
        ax.add_patch(arrow)

    sx, sy = start[1] - 0.5, start[0] - 0.5
    ax.text(sx, sy, "Start", color="darkred", fontsize=13, fontweight="bold", ha="center", va="center", zorder=10)

    for (r, c), k in sorted(bombs.items(), key=lambda kv: kv[1]):
        x, y = c - 0.5, r - 0.5
        ax.text(
            x,
            y,
            f"B{k}",
            color="red",
            fontsize=12,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="circle,pad=0.25", facecolor="yellow", edgecolor="orange", linewidth=2),
            zorder=15,
        )

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    ax.set_xticks([i - 0.5 for i in range(1, ncols + 1)])
    ax.set_xticklabels([str(i) for i in range(1, ncols + 1)], fontsize=12)
    ax.xaxis.tick_top()

    ax.set_yticks([i - 0.5 for i in range(1, nrows + 1)])
    ax.set_yticklabels([str(i) for i in range(1, nrows + 1)], fontsize=12)

    ax.set_title(title, fontsize=16, pad=20)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    return fig


def fig_to_bytes(fig, fmt: str) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format=fmt, dpi=300, bbox_inches="tight")
    buffer.seek(0)
    return buffer.read()




# ============================================================
# Path-planning algorithm
# ============================================================
# Game rule: ONLY these two first moves are allowed.
FIRST_DIRECTIONS = {
    "Up": (-1, 0),
    "Right": (0, 1),
}

# After the first move, normal movement is allowed in all four directions.
MOVE_DIRECTIONS = [
    (-1, 0),  # Up
    (0, 1),   # Right
    (1, 0),   # Down
    (0, -1),  # Left
]


def neighbors(cell: Cell, nrows: int, ncols: int) -> List[Cell]:
    r, c = cell
    out = []
    for dr, dc in MOVE_DIRECTIONS:
        nr, nc = r + dr, c + dc
        if 1 <= nr <= nrows and 1 <= nc <= ncols:
            out.append((nr, nc))
    return out


def first_move_cell(start: Cell, direction: str) -> Cell:
    dr, dc = FIRST_DIRECTIONS[direction]
    return start[0] + dr, start[1] + dc


def infer_first_direction(path: List[Cell]) -> str:
    if len(path) < 2:
        return "None"
    r0, c0 = path[0]
    r1, c1 = path[1]
    delta = (r1 - r0, c1 - c0)
    for name, vec in FIRST_DIRECTIONS.items():
        if delta == vec:
            return name
    return "Invalid"


def plan_longest_path(
    nrows: int,
    ncols: int,
    rocks: Set[Cell],
    start: Cell,
    max_bombs: int = 3,
    time_limit_per_direction: float = 8.0,
    first_move_mode: str = "Auto",
) -> Tuple[List[Cell], Dict[Cell, int], bool, Dict[str, dict]]:
    """
    Longest-simple-path planner with a bomb budget.

    Game constraints
    ----------------
    1. Start from the confirmed start cell.
    2. The FIRST move can only be Up or Right.
    3. After the first move, Up/Right/Down/Left are all allowed.
    4. A cell cannot be visited twice.
    5. Entering a rock cell consumes one bomb.
    6. At most `max_bombs` bombs may be used.
    7. Objective: maximize the number of visited cells.

    Search design
    -------------
    "Auto" solves TWO independent sub-problems:
        A. force first move = Up
        B. force first move = Right
    Each receives the SAME time budget. This prevents one starting direction
    from consuming the search time before the other direction is evaluated.

    Internally the search uses:
    - integer bitmasks for visited cells (faster than copying Python sets);
    - depth-first branch-and-bound;
    - Warnsdorff-style move ordering (constrained cells first);
    - a connectivity upper bound: if the already-used path disconnects the
      remaining board, unreachable cells are excluded from the optimistic bound;
    - memoization of previously explored (current, visited, bombs-used) states.

    Because longest simple path is NP-hard, a time-limited result may still be
    approximate. The app reports results for Up and Right separately so the
    user can see which first move produced the better route.
    """

    total_cells = nrows * ncols

    def idx(cell: Cell) -> int:
        r, c = cell
        return (r - 1) * ncols + (c - 1)

    def cell_of(i: int) -> Cell:
        return (i // ncols + 1, i % ncols + 1)

    # Precompute adjacency for speed.
    adjacency: List[List[int]] = [[] for _ in range(total_cells)]
    for r in range(1, nrows + 1):
        for c in range(1, ncols + 1):
            i = idx((r, c))
            adjacency[i] = [idx(nb) for nb in neighbors((r, c), nrows, ncols)]

    rock_indices = {idx(cell) for cell in rocks}
    rock_mask = 0
    for i in rock_indices:
        rock_mask |= 1 << i

    start_i = idx(start)
    start_is_rock = start_i in rock_indices
    starting_bombs = 1 if start_is_rock else 0
    if starting_bombs > max_bombs:
        return [], {}, True, {}

    # Theoretical maximum number of cells that could possibly be visited.
    non_rock_count = total_cells - len(rock_indices)
    theoretical_max = non_rock_count + min(max_bombs, len(rock_indices))

    def legal_entry(i: int, visited_mask: int, bombs_used: int) -> bool:
        if visited_mask & (1 << i):
            return False
        if i in rock_indices and bombs_used >= max_bombs:
            return False
        return True

    def relaxed_reachable_count(current_i: int, visited_mask: int, bombs_used: int) -> int:
        """
        Connectivity upper bound.

        Count cells that remain connected to the current endpoint after already
        visited cells are blocked. Rock cells are treated optimistically:
        - with zero bombs left, unvisited rocks are blocked;
        - with at least one bomb left, all unvisited rocks are temporarily
          allowed for this bound.

        This deliberately overestimates what is achievable, making it safe for
        branch-and-bound while still detecting regions cut off by the path.
        """
        remaining_bombs = max_bombs - bombs_used
        stack = [current_i]
        seen = {current_i}

        while stack:
            u = stack.pop()
            for v in adjacency[u]:
                bit = 1 << v

                # Other visited cells cannot be crossed.
                if (visited_mask & bit) and v != current_i:
                    continue

                # If no bombs remain, unvisited rock cells cannot be entered.
                if v in rock_indices and not (visited_mask & bit) and remaining_bombs <= 0:
                    continue

                if v not in seen:
                    seen.add(v)
                    stack.append(v)

        # Includes current endpoint.
        return len(seen)

    def onward_degree(i: int, visited_mask: int, bombs_used: int) -> int:
        degree = 0
        for v in adjacency[i]:
            if legal_entry(v, visited_mask, bombs_used):
                degree += 1
        return degree

    def solve_for_first_direction(direction: str) -> dict:
        first_cell = first_move_cell(start, direction)

        # First move must be inside the board.
        if not (1 <= first_cell[0] <= nrows and 1 <= first_cell[1] <= ncols):
            return {
                "direction": direction,
                "path": [start],
                "bombs": {},
                "exact": True,
                "valid": False,
                "reason": "First move is outside the board.",
            }

        first_i = idx(first_cell)
        initial_visited = 1 << start_i

        if not legal_entry(first_i, initial_visited, starting_bombs):
            return {
                "direction": direction,
                "path": [start],
                "bombs": {},
                "exact": True,
                "valid": False,
                "reason": "First move is blocked and cannot be entered with the available bomb budget.",
            }

        extra_first_bomb = 1 if first_i in rock_indices else 0
        bombs_used0 = starting_bombs + extra_first_bomb
        visited0 = initial_visited | (1 << first_i)
        path0 = [start_i, first_i]

        bomb_seq0: List[int] = []
        if start_is_rock:
            bomb_seq0.append(start_i)
        if extra_first_bomb:
            bomb_seq0.append(first_i)

        best_path_i = path0.copy()
        best_bomb_seq_i = bomb_seq0.copy()
        deadline = time.time() + float(time_limit_per_direction)
        timed_out = False

        # Memoization. The value stores the greatest path length seen for the
        # state; because visited_mask already determines path length, duplicate
        # states can simply be skipped.
        seen_states = set()

        def dfs(
            current_i: int,
            visited_mask: int,
            path_i: List[int],
            bombs_used: int,
            bomb_seq_i: List[int],
        ) -> None:
            nonlocal best_path_i, best_bomb_seq_i, timed_out

            if time.time() >= deadline:
                timed_out = True
                return

            state = (current_i, visited_mask, bombs_used)
            if state in seen_states:
                return
            seen_states.add(state)

            current_len = len(path_i)
            if current_len > len(best_path_i):
                best_path_i = path_i.copy()
                best_bomb_seq_i = bomb_seq_i.copy()

                if len(best_path_i) >= theoretical_max:
                    return

            # Connectivity-aware optimistic upper bound.
            reachable_including_current = relaxed_reachable_count(
                current_i, visited_mask, bombs_used
            )
            optimistic_total = current_len - 1 + reachable_including_current
            if optimistic_total <= len(best_path_i):
                return

            candidates = []
            for v in adjacency[current_i]:
                if not legal_entry(v, visited_mask, bombs_used):
                    continue

                extra_bomb = 1 if v in rock_indices else 0
                next_bombs = bombs_used + extra_bomb
                next_visited = visited_mask | (1 << v)

                # Heuristic score:
                # 1) constrained cells first,
                # 2) preserve bombs when otherwise tied,
                # 3) prefer moves retaining a larger connected remainder.
                degree = onward_degree(v, next_visited, next_bombs)
                reachable = relaxed_reachable_count(v, next_visited, next_bombs)
                candidates.append((degree, extra_bomb, -reachable, v))

            candidates.sort()

            for _, extra_bomb, _, v in candidates:
                if timed_out:
                    return

                next_visited = visited_mask | (1 << v)
                path_i.append(v)
                if extra_bomb:
                    bomb_seq_i.append(v)

                dfs(
                    v,
                    next_visited,
                    path_i,
                    bombs_used + extra_bomb,
                    bomb_seq_i,
                )

                if extra_bomb:
                    bomb_seq_i.pop()
                path_i.pop()

                if len(best_path_i) >= theoretical_max:
                    return

        dfs(first_i, visited0, path0, bombs_used0, bomb_seq0)

        best_path = [cell_of(i) for i in best_path_i]
        bomb_order = {
            cell_of(i): order
            for order, i in enumerate(best_bomb_seq_i, start=1)
        }

        return {
            "direction": direction,
            "path": best_path,
            "bombs": bomb_order,
            "exact": not timed_out,
            "valid": True,
            "reason": "",
            "visited": len(best_path),
            "moves": max(0, len(best_path) - 1),
            "bombs_used": len(bomb_order),
            "states_explored": len(seen_states),
        }

    # Only Up and Right are ever considered.
    if first_move_mode == "Up":
        directions_to_solve = ["Up"]
    elif first_move_mode == "Right":
        directions_to_solve = ["Right"]
    else:
        directions_to_solve = ["Up", "Right"]

    results: Dict[str, dict] = {}
    for direction in directions_to_solve:
        results[direction] = solve_for_first_direction(direction)

    valid_results = [r for r in results.values() if r.get("valid", False)]
    if not valid_results:
        return [start], {}, True, results

    # Primary objective: maximum visited cells.
    # Tie-break 1: fewer bombs used (same score with less resource).
    # Tie-break 2: prefer an exact result over a timed-out result.
    # Final tie-break is stable because Up is evaluated before Right.
    winner = max(
        valid_results,
        key=lambda r: (
            r.get("visited", len(r["path"])),
            -r.get("bombs_used", len(r["bombs"])),
            1 if r.get("exact", False) else 0,
        ),
    )

    all_exact = all(r.get("exact", True) for r in valid_results)

    return winner["path"], winner["bombs"], all_exact, results


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="Planting Path Board Generator", layout="wide")
st.title("Planting Path Board Generator with Computer Vision")

st.markdown(
    """
This version adds computer vision recognition before human confirmation:

**Input screenshot → detect board boundary → divide into grid → recognize rock/start cells → user confirms/corrects → generate clean Planting Path Board.**
"""
)

if cv2 is None:
    st.error("OpenCV is not installed. Please install dependencies from the requirements file.")
    st.stop()

if "detected_rocks_text" not in st.session_state:
    st.session_state.detected_rocks_text = cell_to_text(DEFAULT_ROCKS)
if "detected_start_text" not in st.session_state:
    st.session_state.detected_start_text = f"{DEFAULT_START[0]},{DEFAULT_START[1]}"
if "detected_bombs_text" not in st.session_state:
    # Bomb cells should be decided by the path planner, not pre-filled.
    # The user can still manually edit this field after planning.
    st.session_state.detected_bombs_text = ""
if "path_text" not in st.session_state:
    # Keep this empty at startup.
    # It will be filled only after the user clicks "Plan path automatically",
    # or the user can manually paste/type a path.
    st.session_state.path_text = ""

# Apply path-planner updates before the text_area widgets are instantiated.
# Streamlit does not allow modifying st.session_state.<widget_key>
# after that widget has already been created in the same run.
if "pending_planned_path_text" in st.session_state:
    st.session_state.path_text = st.session_state.pop("pending_planned_path_text")
if "pending_planned_bombs_text" in st.session_state:
    st.session_state.detected_bombs_text = st.session_state.pop("pending_planned_bombs_text")


left, middle, right = st.columns([0.95, 1.05, 1.15])

with left:
    st.header("1. Upload screenshot")
    uploaded = st.file_uploader("Original game screenshot", type=["png", "jpg", "jpeg"])

    nrows = st.number_input("Rows", min_value=1, max_value=20, value=7, step=1)
    ncols = st.number_input("Columns", min_value=1, max_value=20, value=7, step=1)

    rock_threshold = st.slider("Rock detection threshold", 0.01, 0.30, 0.08, 0.01)
    start_threshold = st.slider("Start detection threshold", 0.01, 0.30, 0.08, 0.01)
    cell_margin = st.slider("Cell crop margin", 0.00, 0.30, 0.12, 0.01)

    image = None
    auto_bbox = None

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded screenshot", use_container_width=True)
        auto_bbox = find_board_bbox_auto(image)

with middle:
    st.header("2. Computer vision recognition")

    if image is None:
        st.info("Upload a screenshot first.")
    else:
        w, h = image.size

        if auto_bbox is None:
            st.warning("Automatic board-boundary detection failed. Use the sliders below.")
            auto_bbox = (0, 0, min(w, h), min(w, h))
        else:
            st.success(f"Detected board boundary: {auto_bbox}")

        st.subheader("Confirm / correct board boundary")
        x0_default, y0_default, x1_default, y1_default = auto_bbox

        x0 = st.slider("x0", 0, w - 1, int(max(0, min(w - 1, x0_default))))
        y0 = st.slider("y0", 0, h - 1, int(max(0, min(h - 1, y0_default))))
        x1 = st.slider("x1", 1, w, int(max(1, min(w, x1_default))))
        y1 = st.slider("y1", 1, h, int(max(1, min(h, y1_default))))

        bbox = (min(x0, x1 - 1), min(y0, y1 - 1), max(x0 + 1, x1), max(y0 + 1, y1))

        recognize = st.button("Recognize cells from screenshot", type="primary")

        if recognize:
            rocks, start, diagnostics = detect_cells_from_bbox(
                image,
                bbox,
                int(nrows),
                int(ncols),
                rock_threshold=float(rock_threshold),
                start_threshold=float(start_threshold),
                margin=float(cell_margin),
            )

            st.session_state.detected_rocks_text = cell_to_text(rocks)
            if start is not None:
                st.session_state.detected_start_text = f"{start[0]},{start[1]}"

            overlay = draw_detection_overlay(image, bbox, int(nrows), int(ncols), rocks, start)
            st.image(overlay, caption="Detection overlay: orange = rock, green = start", use_container_width=True)

            st.write(f"Detected rocks: {cell_to_text(rocks)}")
            st.write(f"Detected start: {start}")

        else:
            # Show only bbox/grid preview before recognition
            preview = draw_detection_overlay(image, bbox, int(nrows), int(ncols), set(), None)
            st.image(preview, caption="Board boundary/grid preview", use_container_width=True)

with right:
    st.header("3. Human confirmation")
    st.caption("Correct these fields if the computer vision result is not perfect.")

    rocks_text = st.text_area(
        "Confirmed rock cells",
        key="detected_rocks_text",
        height=110,
    )

    start_text = st.text_input(
        "Confirmed start cell",
        key="detected_start_text",
    )

    bombs_text = st.text_area(
        "Bomb cells, generated during planning",
        key="detected_bombs_text",
        height=90,
        placeholder="This will be filled after clicking 'Plan path automatically', e.g. {(7,2): 1, (1,5): 2, (7,7): 3}",
    )

    path_text = st.text_area(
        "Path cells, generated after planning or entered manually",
        key="path_text",
        height=160,
        placeholder="Click 'Plan path automatically' to fill this, or enter a path such as (7,1)->(6,1)->...",
    )

    st.subheader("Automatic path planning")
    st.caption(
        "Game rule: the first move can only be **Up** or **Right**. "
        "After the first move, movement may use all four directions."
    )

    max_bombs = st.number_input(
        "Maximum bombs",
        min_value=0,
        max_value=10,
        value=3,
        step=1,
    )

    solver_time_limit = st.slider(
        "Solver time limit per starting direction, seconds",
        1.0,
        60.0,
        8.0,
        1.0,
        help=(
            "In Auto mode, Up and Right each receive this full time budget. "
            "This makes the comparison fair, but Auto can take up to twice this time."
        ),
    )

    first_move_mode = st.selectbox(
        "First move",
        ["Auto: compare Up and Right", "Up", "Right"],
        index=0,
        help=(
            "Auto solves an Up-first path and a Right-first path independently, "
            "then keeps the one visiting more cells."
        ),
    )

    planner_mode = (
        "Auto"
        if first_move_mode.startswith("Auto")
        else first_move_mode
    )

    if st.button("Plan path automatically"):
        try:
            confirmed_rocks = parse_cell_set(st.session_state.detected_rocks_text)
            confirmed_start = parse_cell(st.session_state.detected_start_text)

            planned_path, planned_bombs, exact, direction_results = plan_longest_path(
                int(nrows),
                int(ncols),
                confirmed_rocks,
                confirmed_start,
                max_bombs=int(max_bombs),
                time_limit_per_direction=float(solver_time_limit),
                first_move_mode=planner_mode,
            )

            if not planned_path or len(planned_path) < 2:
                st.error(
                    "No valid path was found. Check the start cell, rock cells, "
                    "and whether Up or Right can be entered."
                )
            else:
                st.session_state.pending_planned_path_text = path_to_text(planned_path)
                st.session_state.pending_planned_bombs_text = str(planned_bombs)

                # Store a compact comparison so it remains visible after rerun.
                summary = {}
                for direction, result in direction_results.items():
                    summary[direction] = {
                        "valid": result.get("valid", False),
                        "visited": result.get("visited", len(result.get("path", []))),
                        "moves": result.get("moves", max(0, len(result.get("path", [])) - 1)),
                        "bombs_used": result.get("bombs_used", len(result.get("bombs", {}))),
                        "exact": result.get("exact", True),
                        "states_explored": result.get("states_explored", 0),
                        "reason": result.get("reason", ""),
                    }
                st.session_state["last_planner_summary"] = summary
                st.session_state["last_planner_winner"] = infer_first_direction(planned_path)

                st.rerun()

        except Exception as e:
            st.error(f"Path planning failed: {e}")

    if "last_planner_summary" in st.session_state:
        summary = st.session_state["last_planner_summary"]
        winner = st.session_state.get("last_planner_winner", "Unknown")

        st.write(f"**Selected first direction: {winner}**")

        rows = []
        for direction in ["Up", "Right"]:
            if direction not in summary:
                continue
            result = summary[direction]
            rows.append(
                {
                    "First direction": direction,
                    "Valid": result["valid"],
                    "Visited cells": result["visited"],
                    "Moves": result["moves"],
                    "Bombs used": result["bombs_used"],
                    "Exact search": result["exact"],
                    "States explored": result["states_explored"],
                }
            )

        if rows:
            st.dataframe(rows, use_container_width=True, hide_index=True)

        selected = summary.get(winner)
        if selected:
            if selected["exact"]:
                st.success(
                    f"Best route starts {winner}: "
                    f"{selected['visited']} visited cells, "
                    f"{selected['moves']} moves, "
                    f"{selected['bombs_used']} bombs."
                )
            else:
                st.warning(
                    f"Best route found starts {winner}: "
                    f"{selected['visited']} visited cells, "
                    f"{selected['moves']} moves, "
                    f"{selected['bombs_used']} bombs. "
                    "The time limit was reached, so a still-better route may exist."
                )

    st.header("4. Generate clean board")
    generate = st.button("Generate Planting Path Board")

    if generate:
        try:
            rocks = parse_cell_set(rocks_text)
            start = parse_cell(start_text)
            bombs = parse_bombs(bombs_text)

            if not path_text.strip():
                st.error("No path is available yet. Click 'Plan path automatically' first, or manually enter a path.")
                st.stop()

            path = parse_path(path_text)

            warnings = validate_path(path, int(nrows), int(ncols))
            for warning in warnings:
                st.warning(warning)

            rock_cells_in_path = [cell for cell in path if cell in rocks]
            if rock_cells_in_path and not bombs:
                st.warning(
                    "The path passes through rock cells, but no bomb cells are specified. "
                    "Click 'Plan path automatically' to generate bomb cells, or enter them manually."
                )

            fig = render_board(
                int(nrows),
                int(ncols),
                rocks,
                bombs,
                start,
                path,
                title="Planting Path Board",
            )

            st.pyplot(fig, use_container_width=True)

            png_bytes = fig_to_bytes(fig, "png")
            svg_bytes = fig_to_bytes(fig, "svg")

            st.download_button(
                "Download PNG",
                data=png_bytes,
                file_name="planting_path_board.png",
                mime="image/png",
            )

            st.download_button(
                "Download SVG",
                data=svg_bytes,
                file_name="planting_path_board.svg",
                mime="image/svg+xml",
            )

            st.success(f"Visited cells: {len(path)}; moves: {max(0, len(path) - 1)}")

        except Exception as e:
            st.error(f"Could not generate the board: {e}")
