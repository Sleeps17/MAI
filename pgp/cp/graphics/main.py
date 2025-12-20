import matplotlib.pyplot as plt
import numpy as np

# Настройка для красивых графиков
plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["font.size"] = 11

# ============================================================
# ДАННЫЕ ДЛЯ ГРАФИКА 1: Время от размера блока GPU
# ============================================================
# Размеры блока (BLOCK_SIZE x BLOCK_SIZE)
block_sizes = [4, 8, 16, 32, 64]
block_times = [257.574, 69.171, 34.768, 20.199, 7.967]  # Пример

# ============================================================
# ДАННЫЕ ДЛЯ ГРАФИКА 2: CPU vs GPU для разных разрешений
# ============================================================
# Разрешения
resolutions = ["320×240", "640×480", "1280×720", "1920×1080"]
pixels = [76800, 307200, 921600, 2073600]

# Общее время обработки всех кадров (мс)
# ВСТАВЬ СВОИ ДАННЫЕ СЮДА:
cpu_total_times = [2403.605, 9116.903, 26618.397, 60087.798]
gpu_total_times = [83.631, 307.202, 920.384, 2085.804]

# ============================================================
# ПОСТРОЕНИЕ ГРАФИКОВ
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ===== График 1: Зависимость времени от размера блока =====
ax1.plot(block_sizes, block_times, "o-", linewidth=2, markersize=8, color="#2E86AB")
ax1.set_xlabel("Размер блока (BLOCK_SIZE × BLOCK_SIZE)", fontsize=12)
ax1.set_ylabel("Среднее время на кадр (мс)", fontsize=12)
ax1.set_title(
    "Влияние размера блока GPU на производительность\n(1920×1080)",
    fontsize=13,
    fontweight="bold",
)
ax1.grid(True, alpha=0.3, linestyle="--")
ax1.set_xticks(block_sizes)
ax1.set_xticklabels([f"{b}×{b}" for b in block_sizes])

# Добавляем значения на точки
for x, y in zip(block_sizes, block_times):
    ax1.annotate(
        f"{y:.2f} мс",
        xy=(x, y),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.3),
    )

# Оптимальный размер блока
optimal_idx = np.argmin(block_times)
ax1.axvline(
    x=block_sizes[optimal_idx],
    color="red",
    linestyle="--",
    alpha=0.5,
    label="Оптимальный",
)
ax1.legend()

# ===== График 2: CPU vs GPU для разных разрешений =====
x_pos = np.arange(len(resolutions))
width = 0.35

bars1 = ax2.bar(
    x_pos - width / 2, cpu_total_times, width, label="CPU", color="#E63946", alpha=0.8
)
bars2 = ax2.bar(
    x_pos + width / 2, gpu_total_times, width, label="GPU", color="#06D6A0", alpha=0.8
)

ax2.set_xlabel("Разрешение", fontsize=12)
ax2.set_ylabel("Общее время обработки (мс)", fontsize=12)
ax2.set_title(
    "Сравнение CPU vs GPU при разных разрешениях\n(10 кадров для каждого)",
    fontsize=13,
    fontweight="bold",
)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(resolutions, rotation=15)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3, axis="y", linestyle="--")

# Добавляем значения на столбцы
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

# Добавляем ускорение над парами столбцов
for i, (cpu, gpu) in enumerate(zip(cpu_total_times, gpu_total_times)):
    if gpu > 0:
        speedup = cpu / gpu
        y_pos = max(cpu, gpu) * 1.05
        ax2.text(
            i,
            y_pos,
            f"{speedup:.1f}×",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#1D3557",
        )

plt.tight_layout()
plt.savefig("benchmark_results.png", dpi=300, bbox_inches="tight")
print("График сохранён в benchmark_results.png")
plt.show()

# ============================================================
# ДОПОЛНИТЕЛЬНО: Таблица с результатами
# ============================================================
print("\n=== Результаты бенчмарка ===\n")

print("1. Зависимость от размера блока GPU (640×480):")
print("-" * 40)
for bs, time in zip(block_sizes, block_times):
    print(f"  {bs}×{bs:2d} потоков: {time:6.2f} мс")
optimal = block_sizes[optimal_idx]
print(f"\n  Оптимальный размер: {optimal}×{optimal}")

print("\n2. Сравнение CPU vs GPU:")
print("-" * 60)
print(
    f"{'Разрешение':<15} {'Пиксели':<12} {'CPU (мс)':<12} {'GPU (мс)':<12} {'Ускорение'}"
)
print("-" * 60)
for res, pix, cpu, gpu in zip(resolutions, pixels, cpu_total_times, gpu_total_times):
    speedup = cpu / gpu if gpu > 0 else 0
    print(f"{res:<15} {pix:<12} {cpu:<12.1f} {gpu:<12.1f} {speedup:.1f}×")
