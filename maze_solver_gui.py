from __future__ import annotations

import math
import queue
import threading
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


WINDOW_TITLE = "Решатель лабиринтов"
MAX_PROCESSING_SIDE = 1100
PATH_COLOR = (30, 55, 255)
START_COLOR = (60, 190, 60)
END_COLOR = (55, 175, 255)
WAVE_KERNEL = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


@dataclass
class BorderOpening:
    point: tuple[int, int]
    length: int


@dataclass
class CandidateMask:
    name: str
    walkable: np.ndarray
    openings: list[BorderOpening]
    score: float
    walkable_ratio: float
    border_ratio: float


@dataclass
class SolveResult:
    path_points: list[tuple[int, int]]
    start_point: tuple[int, int]
    end_point: tuple[int, int]
    mode_label: str


def read_image(path: Path) -> np.ndarray:
    raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Не удалось открыть изображение.")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    suffix = path.suffix.lower() or ".png"
    if suffix in {".jpg", ".jpeg"}:
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
    else:
        ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError("Не удалось сохранить изображение.")
    path.write_bytes(encoded.tobytes())


def resize_for_processing(image: np.ndarray, max_side: int = MAX_PROCESSING_SIDE) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale >= 1.0:
        return image.copy(), 1.0

    resized = cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def longest_true_run(values: np.ndarray) -> int:
    longest = 0
    current = 0
    for item in values:
        if bool(item):
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return longest


def group_consecutive(indices: np.ndarray) -> list[np.ndarray]:
    if len(indices) == 0:
        return []

    groups: list[list[int]] = [[int(indices[0])]]
    for index in indices[1:]:
        value = int(index)
        if value == groups[-1][-1] + 1:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [np.array(group, dtype=np.int32) for group in groups]


def detect_maze_frame(gray: np.ndarray) -> tuple[int, int, int, int] | None:
    height, width = gray.shape
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    dark_threshold = min(170, int(np.percentile(blurred, 22)))
    dark_threshold = max(55, dark_threshold)

    dark_mask = blurred <= dark_threshold
    dark_mask = cv2.dilate(dark_mask.astype(np.uint8) * 255, np.ones((3, 3), dtype=np.uint8), iterations=1) > 0

    row_runs = np.array([longest_true_run(row) for row in dark_mask], dtype=np.int32)
    col_runs = np.array([longest_true_run(col) for col in dark_mask.T], dtype=np.int32)

    min_horizontal_run = max(80, int(width * 0.5))
    min_vertical_run = max(80, int(height * 0.5))
    edge_margin_y = max(3, height // 120)
    edge_margin_x = max(3, width // 120)

    candidate_rows = np.where(row_runs >= min_horizontal_run)[0]
    candidate_rows = candidate_rows[(candidate_rows >= edge_margin_y) & (candidate_rows < height - edge_margin_y)]
    candidate_cols = np.where(col_runs >= min_vertical_run)[0]
    candidate_cols = candidate_cols[(candidate_cols >= edge_margin_x) & (candidate_cols < width - edge_margin_x)]

    row_groups = group_consecutive(candidate_rows)
    col_groups = group_consecutive(candidate_cols)

    top_candidates = [group for group in row_groups if int(np.median(group)) < height * 0.4]
    bottom_candidates = [group for group in row_groups if int(np.median(group)) > height * 0.6]
    left_candidates = [group for group in col_groups if int(np.median(group)) < width * 0.4]
    right_candidates = [group for group in col_groups if int(np.median(group)) > width * 0.6]

    if not top_candidates or not bottom_candidates or not left_candidates or not right_candidates:
        return None

    top = int(np.median(top_candidates[0]))
    bottom = int(np.median(bottom_candidates[-1]))
    left = int(np.median(left_candidates[0]))
    right = int(np.median(right_candidates[-1]))

    if right - left < width * 0.45 or bottom - top < height * 0.45:
        return None

    return left, top, right, bottom


def crop_to_content(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    frame_bbox = detect_maze_frame(gray)
    if frame_bbox is not None:
        left, top, right, bottom = frame_bbox
        cropped = image[top : bottom + 1, left : right + 1].copy()
        if cropped.size > 0:
            return cropped, (left, top)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    crop_masks = [
        otsu == 0,
        otsu > 0,
    ]

    best_bbox: tuple[int, int, int, int] | None = None
    best_ratio = 1.0
    for mask in crop_masks:
        ratio = float(mask.mean())
        if not 0.01 <= ratio <= 0.68:
            continue
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            continue
        if ratio < best_ratio:
            best_ratio = ratio
            best_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    if best_bbox is None:
        band = max(3, min(height, width) // 45)
        border_values = np.concatenate(
            [
                blur[:band, :].ravel(),
                blur[-band:, :].ravel(),
                blur[:, :band].ravel(),
                blur[:, -band:].ravel(),
            ]
        )
        background_level = int(np.median(border_values))
        threshold = max(14, int(np.std(border_values) * 1.8) + 8)
        diff = cv2.absdiff(blur, np.full_like(blur, background_level))
        fallback_mask = diff > threshold
        ys, xs = np.where(fallback_mask)
        if len(xs) == 0 or len(ys) == 0:
            return image.copy(), (0, 0)
        best_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    left, top, right, bottom = best_bbox
    cropped = image[top : bottom + 1, left : right + 1].copy()
    if cropped.size == 0:
        return image.copy(), (0, 0)
    return cropped, (left, top)


def border_points(width: int, height: int) -> list[tuple[int, int]]:
    points = [(x, 0) for x in range(width)]
    points.extend((width - 1, y) for y in range(1, height))
    if height > 1:
        points.extend((x, height - 1) for x in range(width - 2, -1, -1))
    if width > 1:
        points.extend((0, y) for y in range(height - 2, 0, -1))
    return points


def find_border_openings(mask: np.ndarray) -> tuple[list[BorderOpening], float]:
    height, width = mask.shape
    ring = border_points(width, height)
    if not ring:
        return [], 0.0

    values = [bool(mask[y, x]) for x, y in ring]
    perimeter = len(values)
    open_count = sum(values)
    if open_count == 0:
        return [], 0.0

    runs: list[tuple[int, int]] = []
    start = None
    for index, is_open in enumerate(values):
        if is_open and start is None:
            start = index
        elif not is_open and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, perimeter - 1))

    if len(runs) > 1 and values[0] and values[-1]:
        first_start, first_end = runs[0]
        last_start, last_end = runs[-1]
        merged = (last_start, first_end + perimeter)
        runs = [merged] + runs[1:-1]

    openings: list[BorderOpening] = []
    for run_start, run_end in runs:
        center_index = (run_start + run_end) // 2
        x, y = ring[center_index % perimeter]
        openings.append(BorderOpening(point=(x, y), length=run_end - run_start + 1))

    border_ratio = open_count / perimeter
    return openings, border_ratio


def score_candidate(mask: np.ndarray) -> CandidateMask:
    walkable_ratio = float(mask.mean())
    openings, border_ratio = find_border_openings(mask)
    opening_lengths = [opening.length for opening in openings]
    widest_ratio = (max(opening_lengths) / max(1, len(border_points(mask.shape[1], mask.shape[0])))) if opening_lengths else 0.0

    score = 0.0
    score -= abs(walkable_ratio - 0.42) * 3.0
    if 2 <= len(openings) <= 8:
        score += 2.0
    elif len(openings) == 1:
        score += 0.2
    else:
        score -= min(1.5, len(openings) * 0.18)
    score -= max(0.0, border_ratio - 0.25) * 4.0
    score -= widest_ratio * 4.0

    return CandidateMask(
        name="",
        walkable=mask,
        openings=openings,
        score=score,
        walkable_ratio=walkable_ratio,
        border_ratio=border_ratio,
    )


def build_candidate_masks(image: np.ndarray) -> list[CandidateMask]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    min_side = min(blurred.shape[:2])
    block_size = max(15, min(71, (min_side // 14) | 1))
    if block_size % 2 == 0:
        block_size += 1

    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        7,
    )
    otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    variants = [
        ("Адаптивный порог", adaptive),
        ("Адаптивный инвертированный", 255 - adaptive),
        ("Оцу", otsu),
        ("Оцу инвертированный", 255 - otsu),
    ]

    candidates: list[CandidateMask] = []
    for name, binary in variants:
        cleaned = cv2.medianBlur(binary, 3)
        mask = cleaned > 0
        candidate = score_candidate(mask)
        candidate.name = name
        candidates.append(candidate)

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


def snap_to_walkable(mask: np.ndarray, point: tuple[int, int], radius: int = 18) -> tuple[int, int] | None:
    x, y = point
    height, width = mask.shape
    x = int(np.clip(x, 0, width - 1))
    y = int(np.clip(y, 0, height - 1))
    if mask[y, x]:
        return x, y

    for current_radius in range(1, radius + 1):
        x0 = max(0, x - current_radius)
        x1 = min(width, x + current_radius + 1)
        y0 = max(0, y - current_radius)
        y1 = min(height, y + current_radius + 1)
        region = mask[y0:y1, x0:x1]
        ys, xs = np.where(region)
        if len(xs) == 0:
            continue
        xs = xs + x0
        ys = ys + y0
        distances = (xs - x) ** 2 + (ys - y) ** 2
        index = int(np.argmin(distances))
        return int(xs[index]), int(ys[index])
    return None


def wavefront_shortest_path(
    walkable_mask: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]] | None:
    width = walkable_mask.shape[1]
    height = walkable_mask.shape[0]
    sx, sy = start
    ex, ey = end

    if not (0 <= sx < width and 0 <= sy < height and 0 <= ex < width and 0 <= ey < height):
        return None
    if not walkable_mask[sy, sx] or not walkable_mask[ey, ex]:
        return None

    walkable_u8 = (walkable_mask.astype(np.uint8)) * 255
    frontier = np.zeros((height, width), dtype=np.uint8)
    visited = np.zeros((height, width), dtype=np.uint8)
    distance = np.full((height, width), -1, dtype=np.int32)

    frontier[sy, sx] = 255
    visited[sy, sx] = 255
    distance[sy, sx] = 0

    step = 0
    while frontier.any():
        if visited[ey, ex]:
            break

        expanded = cv2.dilate(frontier, WAVE_KERNEL, iterations=1)
        new_frontier = cv2.bitwise_and(expanded, walkable_u8)
        new_frontier = cv2.bitwise_and(new_frontier, cv2.bitwise_not(visited))
        if not new_frontier.any():
            break

        step += 1
        distance[new_frontier > 0] = step
        visited = cv2.bitwise_or(visited, new_frontier)
        frontier = new_frontier

    if distance[ey, ex] < 0:
        return None

    path: list[tuple[int, int]] = [(ex, ey)]
    current_x, current_y = ex, ey
    current_distance = distance[ey, ex]

    while current_distance > 0:
        moved = False
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_x = current_x + dx
            next_y = current_y + dy
            if 0 <= next_x < width and 0 <= next_y < height and distance[next_y, next_x] == current_distance - 1:
                path.append((next_x, next_y))
                current_x, current_y = next_x, next_y
                current_distance -= 1
                moved = True
                break
        if not moved:
            return None

    path.reverse()
    return path


def simplify_path(path: list[tuple[int, int]], stride: int = 12) -> list[tuple[int, int]]:
    if len(path) <= 2:
        return path[:]

    simplified = [path[0]]
    steps_since_keep = 0
    previous_direction = (path[1][0] - path[0][0], path[1][1] - path[0][1])

    for index in range(1, len(path) - 1):
        current = path[index]
        next_point = path[index + 1]
        current_direction = (next_point[0] - current[0], next_point[1] - current[1])
        steps_since_keep += abs(current[0] - path[index - 1][0]) + abs(current[1] - path[index - 1][1])
        is_turn = current_direction != previous_direction
        if is_turn or steps_since_keep >= stride:
            if current != simplified[-1]:
                simplified.append(current)
            steps_since_keep = 0
        previous_direction = current_direction

    if path[-1] != simplified[-1]:
        simplified.append(path[-1])
    return simplified


def scale_point(point: tuple[int, int], factor: float, width: int, height: int) -> tuple[int, int]:
    x = int(round(point[0] * factor))
    y = int(round(point[1] * factor))
    return int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))


def unscale_points(
    points: list[tuple[int, int]],
    factor: float,
    width: int,
    height: int,
) -> list[tuple[int, int]]:
    if factor == 1.0:
        return points

    output: list[tuple[int, int]] = []
    for x, y in points:
        original_x = int(round(x / factor))
        original_y = int(round(y / factor))
        restored = (
            int(np.clip(original_x, 0, width - 1)),
            int(np.clip(original_y, 0, height - 1)),
        )
        if not output or output[-1] != restored:
            output.append(restored)
    return output


def generate_auto_pairs(openings: list[BorderOpening]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(openings) < 2:
        return []
    pairs = list(combinations(openings, 2))
    pairs.sort(
        key=lambda pair: math.dist(pair[0].point, pair[1].point) + min(pair[0].length, pair[1].length) * 0.2,
        reverse=True,
    )
    return [(first.point, second.point) for first, second in pairs[:12]]


def solve_maze(image: np.ndarray, manual_points: tuple[tuple[int, int], tuple[int, int]] | None = None) -> SolveResult:
    cropped, offset = crop_to_content(image)
    offset_x, offset_y = offset
    cropped_height, cropped_width = cropped.shape[:2]
    processed, factor = resize_for_processing(cropped)
    candidates = build_candidate_masks(processed)
    original_height, original_width = image.shape[:2]
    processed_height, processed_width = processed.shape[:2]

    if manual_points is not None:
        translated_start = (
            int(np.clip(manual_points[0][0] - offset_x, 0, cropped_width - 1)),
            int(np.clip(manual_points[0][1] - offset_y, 0, cropped_height - 1)),
        )
        translated_end = (
            int(np.clip(manual_points[1][0] - offset_x, 0, cropped_width - 1)),
            int(np.clip(manual_points[1][1] - offset_y, 0, cropped_height - 1)),
        )
        scaled_start = scale_point(translated_start, factor, processed_width, processed_height)
        scaled_end = scale_point(translated_end, factor, processed_width, processed_height)
    else:
        scaled_start = None
        scaled_end = None

    for candidate in candidates:
        if manual_points is not None:
            start = snap_to_walkable(candidate.walkable, scaled_start)
            end = snap_to_walkable(candidate.walkable, scaled_end)
            if start is None or end is None:
                continue
            path = wavefront_shortest_path(candidate.walkable, start, end)
            if path is None:
                continue
            simplified = simplify_path(path)
            path_original = [
                (
                    int(np.clip(x + offset_x, 0, original_width - 1)),
                    int(np.clip(y + offset_y, 0, original_height - 1)),
                )
                for x, y in unscale_points(simplified, factor, cropped_width, cropped_height)
            ]
            start_scaled = unscale_points([start], factor, cropped_width, cropped_height)[0]
            end_scaled = unscale_points([end], factor, cropped_width, cropped_height)[0]
            start_original = (
                int(np.clip(start_scaled[0] + offset_x, 0, original_width - 1)),
                int(np.clip(start_scaled[1] + offset_y, 0, original_height - 1)),
            )
            end_original = (
                int(np.clip(end_scaled[0] + offset_x, 0, original_width - 1)),
                int(np.clip(end_scaled[1] + offset_y, 0, original_height - 1)),
            )
            return SolveResult(
                path_points=path_original,
                start_point=start_original,
                end_point=end_original,
                mode_label=f"ручной режим, {candidate.name.lower()}",
            )

        if candidate.border_ratio > 0.42 or candidate.walkable_ratio < 0.04 or candidate.walkable_ratio > 0.88:
            continue

        for start, end in generate_auto_pairs(candidate.openings):
            path = wavefront_shortest_path(candidate.walkable, start, end)
            if path is None:
                continue
            simplified = simplify_path(path)
            path_original = [
                (
                    int(np.clip(x + offset_x, 0, original_width - 1)),
                    int(np.clip(y + offset_y, 0, original_height - 1)),
                )
                for x, y in unscale_points(simplified, factor, cropped_width, cropped_height)
            ]
            start_scaled = unscale_points([start], factor, cropped_width, cropped_height)[0]
            end_scaled = unscale_points([end], factor, cropped_width, cropped_height)[0]
            start_original = (
                int(np.clip(start_scaled[0] + offset_x, 0, original_width - 1)),
                int(np.clip(start_scaled[1] + offset_y, 0, original_height - 1)),
            )
            end_original = (
                int(np.clip(end_scaled[0] + offset_x, 0, original_width - 1)),
                int(np.clip(end_scaled[1] + offset_y, 0, original_height - 1)),
            )
            return SolveResult(
                path_points=path_original,
                start_point=start_original,
                end_point=end_original,
                mode_label=f"автоопределение, {candidate.name.lower()}",
            )

    if manual_points is not None:
        raise ValueError(
            "Не нашел маршрут между выбранными точками. Попробуй выбрать старт и финиш точнее внутри проходов."
        )
    raise ValueError(
        "Автоматически не удалось определить вход и выход. Кликни старт и финиш на картинке и нажми «Найти путь» еще раз."
    )


class MazeSolverApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("1280x860")
        self.root.minsize(920, 640)

        self.image_path: Path | None = None
        self.original_image: np.ndarray | None = None
        self.result_image: np.ndarray | None = None
        self.current_path: list[tuple[int, int]] = []
        self.start_point: tuple[int, int] | None = None
        self.end_point: tuple[int, int] | None = None
        self.display_scale = 1.0
        self.display_offset = (0, 0)
        self.photo_image: ImageTk.PhotoImage | None = None
        self.queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.busy = False
        self.animation_index = 0
        self.animation_points: list[tuple[int, int]] = []
        self.path_thickness = 4

        self.status_var = tk.StringVar(
            value="Открой изображение лабиринта. Если авто-режим не справится, кликни старт и финиш на картинке."
        )

        self.build_ui()
        self.root.after(120, self.process_queue)

    def build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=14)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(container)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        for column in range(6):
            toolbar.columnconfigure(column, weight=0)
        toolbar.columnconfigure(5, weight=1)

        self.open_button = ttk.Button(toolbar, text="Открыть изображение", command=self.open_image)
        self.open_button.grid(row=0, column=0, padx=(0, 8))

        self.solve_button = ttk.Button(toolbar, text="Найти путь", command=self.solve_current_image)
        self.solve_button.grid(row=0, column=1, padx=(0, 8))

        self.reset_button = ttk.Button(toolbar, text="Сбросить точки", command=self.reset_points)
        self.reset_button.grid(row=0, column=2, padx=(0, 8))

        self.save_button = ttk.Button(toolbar, text="Сохранить результат", command=self.save_image)
        self.save_button.grid(row=0, column=3, padx=(0, 8))

        tip = ttk.Label(
            toolbar,
            text="ЛКМ: выбрать старт и финиш. Третий клик начинает выбор заново.",
        )
        tip.grid(row=0, column=5, sticky="e")

        canvas_frame = ttk.Frame(container)
        canvas_frame.grid(row=1, column=0, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            canvas_frame,
            background="#161616",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Configure>", lambda _event: self.refresh_preview())

        status = ttk.Label(
            container,
            textvariable=self.status_var,
            anchor="w",
            relief="groove",
            padding=(10, 8),
        )
        status.grid(row=2, column=0, sticky="ew", pady=(12, 0))

        self.draw_placeholder()

    def set_busy(self, busy: bool) -> None:
        self.busy = busy
        state = "disabled" if busy else "normal"
        self.open_button.configure(state=state)
        self.solve_button.configure(state=state)
        self.reset_button.configure(state=state)
        self.save_button.configure(state=state)

    def draw_placeholder(self) -> None:
        self.canvas.delete("all")
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.canvas.create_text(
            width // 2,
            height // 2,
            text="Здесь появится лабиринт",
            fill="#f0f0f0",
            font=("Segoe UI", 22, "bold"),
        )
        self.canvas.create_text(
            width // 2,
            height // 2 + 34,
            text="Открой файл, затем нажми «Найти путь»",
            fill="#c7c7c7",
            font=("Segoe UI", 12),
        )

    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Выбери изображение лабиринта",
            filetypes=[
                ("Изображения", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("Все файлы", "*.*"),
            ],
        )
        if not path:
            return

        try:
            image = read_image(Path(path))
        except Exception as error:
            messagebox.showerror("Ошибка", str(error))
            return

        self.image_path = Path(path)
        self.original_image = image
        self.result_image = image.copy()
        self.current_path = []
        self.start_point = None
        self.end_point = None
        self.path_thickness = max(3, min(image.shape[:2]) // 120)
        self.status_var.set("Изображение загружено. Можно сразу запускать авто-поиск или выбрать старт и финиш вручную.")
        self.refresh_preview()

    def save_image(self) -> None:
        if self.result_image is None:
            messagebox.showinfo("Нет результата", "Сначала открой и реши лабиринт.")
            return

        suggested_name = "maze_solved.png"
        if self.image_path is not None:
            suggested_name = f"{self.image_path.stem}_solved.png"

        path = filedialog.asksaveasfilename(
            title="Сохранить результат",
            defaultextension=".png",
            initialfile=suggested_name,
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("BMP", "*.bmp"),
                ("Все файлы", "*.*"),
            ],
        )
        if not path:
            return

        try:
            write_image(Path(path), self.compose_display_image())
        except Exception as error:
            messagebox.showerror("Ошибка сохранения", str(error))
            return

        self.status_var.set(f"Результат сохранен: {path}")

    def reset_points(self) -> None:
        if self.original_image is None:
            return
        self.current_path = []
        self.start_point = None
        self.end_point = None
        self.result_image = self.original_image.copy()
        self.status_var.set("Точки сброшены. Можно снова выбрать старт и финиш или запустить авто-поиск.")
        self.refresh_preview()

    def image_coordinates_from_event(self, event: tk.Event) -> tuple[int, int] | None:
        if self.original_image is None:
            return None

        x = int((event.x - self.display_offset[0]) / self.display_scale)
        y = int((event.y - self.display_offset[1]) / self.display_scale)
        height, width = self.original_image.shape[:2]
        if not (0 <= x < width and 0 <= y < height):
            return None
        return x, y

    def on_canvas_click(self, event: tk.Event) -> None:
        if self.original_image is None or self.busy:
            return

        point = self.image_coordinates_from_event(event)
        if point is None:
            return

        self.current_path = []
        self.result_image = self.original_image.copy()

        if self.start_point is None or (self.start_point is not None and self.end_point is not None):
            self.start_point = point
            self.end_point = None
            self.status_var.set("Старт выбран. Теперь кликни на финиш.")
        else:
            self.end_point = point
            self.status_var.set("Финиш выбран. Нажми «Найти путь».")

        self.refresh_preview()

    def solve_current_image(self) -> None:
        if self.original_image is None:
            messagebox.showinfo("Нет изображения", "Сначала открой файл с лабиринтом.")
            return
        if self.busy:
            return
        if (self.start_point is None) ^ (self.end_point is None):
            messagebox.showinfo("Не хватает точки", "Нужно выбрать и старт, и финиш, или не выбирать точки совсем.")
            return

        self.result_image = self.original_image.copy()
        self.current_path = []
        self.refresh_preview()
        self.set_busy(True)
        self.status_var.set("Распознаю проходы и строю маршрут...")

        manual_points = None
        if self.start_point is not None and self.end_point is not None:
            manual_points = (self.start_point, self.end_point)

        thread = threading.Thread(target=self.solve_worker, args=(manual_points,), daemon=True)
        thread.start()

    def solve_worker(self, manual_points: tuple[tuple[int, int], tuple[int, int]] | None) -> None:
        try:
            assert self.original_image is not None
            result = solve_maze(self.original_image, manual_points)
        except Exception as error:
            self.queue.put(("error", str(error)))
            return
        self.queue.put(("solved", result))

    def process_queue(self) -> None:
        try:
            while True:
                event_name, payload = self.queue.get_nowait()
                if event_name == "error":
                    self.set_busy(False)
                    self.status_var.set(str(payload))
                    messagebox.showinfo("Не получилось", str(payload))
                elif event_name == "solved":
                    assert isinstance(payload, SolveResult)
                    self.apply_solution(payload)
        except queue.Empty:
            pass
        finally:
            self.root.after(120, self.process_queue)

    def apply_solution(self, result: SolveResult) -> None:
        if self.original_image is None:
            self.set_busy(False)
            return

        self.start_point = result.start_point
        self.end_point = result.end_point
        self.current_path = result.path_points
        self.result_image = self.original_image.copy()
        self.animation_points = result.path_points
        self.animation_index = 1
        self.status_var.set(f"Маршрут найден: {result.mode_label}. Рисую путь...")
        self.animate_path(result.mode_label)

    def animate_path(self, mode_label: str) -> None:
        if self.result_image is None or len(self.animation_points) < 2:
            self.set_busy(False)
            return

        for _ in range(6):
            if self.animation_index >= len(self.animation_points):
                self.refresh_preview()
                self.set_busy(False)
                self.status_var.set(f"Готово: {mode_label}. Можно сохранить результат или выбрать другие точки.")
                return
            point_a = self.animation_points[self.animation_index - 1]
            point_b = self.animation_points[self.animation_index]
            cv2.line(
                self.result_image,
                point_a,
                point_b,
                PATH_COLOR,
                self.path_thickness,
                lineType=cv2.LINE_AA,
            )
            self.animation_index += 1

        self.refresh_preview()
        self.root.after(18, lambda: self.animate_path(mode_label))

    def compose_display_image(self) -> np.ndarray:
        if self.original_image is None:
            raise ValueError("Нет изображения для отображения.")

        base = self.result_image.copy() if self.result_image is not None else self.original_image.copy()

        marker_radius = max(7, self.path_thickness + 3)
        marker_thickness = max(2, self.path_thickness // 2)

        if self.start_point is not None:
            cv2.circle(base, self.start_point, marker_radius, START_COLOR, -1, lineType=cv2.LINE_AA)
            cv2.circle(base, self.start_point, marker_radius + 2, (255, 255, 255), marker_thickness, lineType=cv2.LINE_AA)

        if self.end_point is not None:
            cv2.circle(base, self.end_point, marker_radius, END_COLOR, -1, lineType=cv2.LINE_AA)
            cv2.circle(base, self.end_point, marker_radius + 2, (255, 255, 255), marker_thickness, lineType=cv2.LINE_AA)

        return base

    def refresh_preview(self) -> None:
        if self.original_image is None:
            self.draw_placeholder()
            return

        canvas_width = max(100, self.canvas.winfo_width())
        canvas_height = max(100, self.canvas.winfo_height())
        image = self.compose_display_image()
        height, width = image.shape[:2]

        scale = min((canvas_width - 20) / width, (canvas_height - 20) / height)
        scale = max(scale, 0.05)
        display_width = max(1, int(width * scale))
        display_height = max(1, int(height * scale))

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb).resize((display_width, display_height), Image.Resampling.LANCZOS)

        self.photo_image = ImageTk.PhotoImage(pil_image)
        self.display_scale = scale
        self.display_offset = ((canvas_width - display_width) // 2, (canvas_height - display_height) // 2)

        self.canvas.delete("all")
        self.canvas.create_image(
            self.display_offset[0],
            self.display_offset[1],
            anchor="nw",
            image=self.photo_image,
        )


def main() -> None:
    root = tk.Tk()
    app = MazeSolverApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
