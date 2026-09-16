from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "learning_curves_raw.png"
EXPERIMENTS = {
    "4 clients - all": ROOT / "experiments/optimized_fedadagrad_4all_20260916_run1_all_4of4/metrics/fl_metrics_fedadagrad_4_train_optimized_fedadagrad_4all_20260916_run1_all_4of4.csv",
    "8 clients - error-low 4/8": ROOT / "experiments/fast_parallel_fedadagrad_8errorlow4_20260916_run1_error_low_4of8/metrics/fl_metrics_fedadagrad_8_train_fast_parallel_fedadagrad_8errorlow4_20260916_run1_error_low_4of8.csv",
    "8 clients - random 4/8": ROOT / "experiments/fast_parallel_fedadagrad_8random4_20260916_run1_random_4of8/metrics/fl_metrics_fedadagrad_8_train_fast_parallel_fedadagrad_8random4_20260916_run1_random_4of8.csv",
}
COLORS = [(51, 102, 204), (220, 57, 18), (16, 150, 24)]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)


def load_means(path: Path) -> list[float]:
    values: dict[int, list[float]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"episode", "path_id", "avg_lateral_error"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: missing required columns")
        for row in reader:
            values[int(row["episode"])].append(float(row["avg_lateral_error"]))
    if sorted(values) != list(range(1000)):
        raise ValueError(f"{path}: expected complete episodes 0..999")
    return [sum(values[i]) / len(values[i]) for i in range(1000)]


def draw_panel(draw, image, box, data, title, x_min, x_max, y_min, y_max, log_y):
    left, top, right, bottom = box
    plot_left, plot_top = left + 105, top + 55
    plot_right, plot_bottom = right - 28, bottom - 68
    axis_color, grid_color = (45, 45, 45), (218, 218, 218)
    draw.text((left, top), title, fill=(25, 25, 25), font=font(24, True))
    draw.rectangle((plot_left, plot_top, plot_right, plot_bottom), outline=(175, 175, 175), width=1)
    x_ticks = [x_min + round((x_max - x_min) * i / 5) for i in range(6)]
    if log_y:
        y_ticks = [0.25, 0.5, 1, 2, 5, 10, 20, 50]
        transform = math.log10
    else:
        y_ticks = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
        transform = lambda value: value

    def sx(value):
        return plot_left + (value - x_min) / (x_max - x_min) * (plot_right - plot_left)

    def sy(value):
        lo, hi = transform(y_min), transform(y_max)
        clipped = min(max(value, y_min), y_max)
        return plot_bottom - (transform(clipped) - lo) / (hi - lo) * (plot_bottom - plot_top)

    for value in x_ticks:
        x = sx(value)
        draw.line((x, plot_top, x, plot_bottom), fill=grid_color, width=1)
        label = str(value)
        bbox = draw.textbbox((0, 0), label, font=font(15))
        draw.text((x - (bbox[2] - bbox[0]) / 2, plot_bottom + 10), label, fill=axis_color, font=font(15))
    for value in y_ticks:
        if y_min <= value <= y_max:
            y = sy(value)
            draw.line((plot_left, y, plot_right, y), fill=grid_color, width=1)
            label = f"{value:g}"
            bbox = draw.textbbox((0, 0), label, font=font(15))
            draw.text((plot_left - 12 - (bbox[2] - bbox[0]), y - 8), label, fill=axis_color, font=font(15))
    for (label, values), color in zip(data.items(), COLORS):
        points = [(sx(ep), sy(values[ep])) for ep in range(x_min, x_max + 1)]
        draw.line(points, fill=color, width=2, joint="curve")
    draw.text(((plot_left + plot_right) / 2 - 28, plot_bottom + 39), "Episode", fill=axis_color, font=font(17))
    y_label = "Mean lateral error" + (" (log scale)" if log_y else "")
    label_img = Image.new("RGBA", (300, 30), (255, 255, 255, 0))
    ImageDraw.Draw(label_img).text((0, 2), y_label, fill=axis_color, font=font(17))
    label_img = label_img.rotate(90, expand=True)
    image.paste(label_img, (left + 12, int((plot_top + plot_bottom) / 2 - label_img.height / 2)), label_img)


data = {label: load_means(path) for label, path in EXPERIMENTS.items()}
image = Image.new("RGB", (1600, 1220), "white")
draw = ImageDraw.Draw(image)
draw.text((70, 28), "FedAdagrad raw training curves", fill=(20, 20, 20), font=font(32, True))
draw.text((70, 72), "Per-episode mean across agents; no smoothing", fill=(80, 80, 80), font=font(18))
legend_x, legend_y = 760, 42
for index, (label, color) in enumerate(zip(data, COLORS)):
    y = legend_y + index * 28
    draw.line((legend_x, y + 9, legend_x + 35, y + 9), fill=color, width=4)
    draw.text((legend_x + 46, y), label, fill=(40, 40, 40), font=font(16))
draw_panel(draw, image, (55, 125, 1545, 650), data, "Full training run", 0, 999, 0.2, 100, True)
draw_panel(draw, image, (55, 675, 1545, 1195), data, "Post-convergence detail", 100, 999, 0.25, 2.0, False)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
image.save(OUTPUT, optimize=True)
print(OUTPUT)
