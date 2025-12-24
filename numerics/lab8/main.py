import time

import matplotlib.pyplot as plt
import numpy as np

# ================= ПАРАМЕТРЫ ЗАДАЧИ =================
a = 1.0
Lx = np.pi / 4
Ly = np.log(2)
T_final = 1.0


def u_exact(x, y, t):
    """Точное аналитическое решение"""
    return np.cos(2 * x) * np.cosh(y) * np.exp(-3 * a * t)


def thomas_algorithm(a, b, c, d):
    n = len(d)

    cp = np.zeros(n)
    dp = np.zeros(n)

    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]

    for i in range(1, n - 1):
        cp[i] = c[i] / (b[i] - a[i] * cp[i - 1])
        dp[i] = (d[i] - a[i] * dp[i - 1]) / (b[i] - a[i] * cp[i - 1])

    dp[n - 1] = (d[n - 1] - a[n - 1] * dp[n - 2]) / (b[n - 1] - a[n - 1] * cp[n - 2])

    x = np.zeros(n)
    x[n - 1] = dp[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x


def solve_ADI(I, J, K):
    print("\nМЕТОД ПЕРЕМЕННЫХ НАПРАВЛЕНИЙ (ADI)")
    print(f"Сетка: {I}×{J} точек, {K} шагов по времени")

    hx = Lx / I
    hy = Ly / J
    tau = T_final / K

    x = np.linspace(0, Lx, I + 1)
    y = np.linspace(0, Ly, J + 1)
    t = np.linspace(0, T_final, K + 1)

    u = np.zeros((I + 1, J + 1, K + 1))

    for i in range(I + 1):
        for j in range(J + 1):
            u[i, j, 0] = np.cos(2 * x[i]) * np.cosh(y[j])

    for k in range(K + 1):
        tk = t[k]
        exp_factor = np.exp(-3 * a * tk)
        u[0, :, k] = np.cosh(y) * exp_factor
        u[I, :, k] = 0.0
        u[:, 0, k] = np.cos(2 * x) * exp_factor

    sigma_x = a * tau / (2 * hx**2)
    sigma_y = a * tau / (2 * hy**2)

    u_star = np.zeros((I + 1, J + 1))

    start_time = time.time()

    for k in range(K):
        t_mid = t[k] + tau / 2
        t_next = t[k + 1]
        exp_mid = np.exp(-3 * a * t_mid)
        exp_next = np.exp(-3 * a * t_next)

        # === ПЕРВЫЙ ПОЛУШАГ: неявно по x, явно по y ===
        for j in range(1, J):
            A = np.zeros(I + 1)
            B = np.zeros(I + 1)
            C = np.zeros(I + 1)
            D = np.zeros(I + 1)

            for i in range(I + 1):
                if i == 0:
                    B[i] = 1.0
                    D[i] = np.cosh(y[j]) * exp_mid
                elif i == I:
                    B[i] = 1.0
                    D[i] = 0.0
                else:
                    A[i] = -sigma_x
                    B[i] = 1.0 + 2 * sigma_x
                    C[i] = -sigma_x
                    D[i] = (
                        sigma_y * u[i, j + 1, k]
                        + (1 - 2 * sigma_y) * u[i, j, k]
                        + sigma_y * u[i, j - 1, k]
                    )

            u_star[:, j] = thomas_algorithm(A, B, C, D)

        u_star[:, 0] = np.cos(2 * x) * exp_mid

        # === ВТОРОЙ ПОЛУШАГ: явно по x, неявно по y ===
        for i in range(1, I):  # внутренние узлы по x
            A = np.zeros(J + 1)
            B = np.zeros(J + 1)
            C = np.zeros(J + 1)
            D = np.zeros(J + 1)

            for j in range(J + 1):
                if j == 0:
                    B[j] = 1.0
                    D[j] = np.cos(2 * x[i]) * exp_next
                elif j == J:
                    B[j] = 3.0
                    A[j] = -3.0
                    C[j] = 0.0
                    D[j] = -hy * np.cos(2 * x[i]) * exp_next
                else:
                    A[j] = -sigma_y
                    B[j] = 1.0 + 2 * sigma_y
                    C[j] = -sigma_y
                    D[j] = (
                        sigma_x * u_star[i + 1, j]
                        + (1 - 2 * sigma_x) * u_star[i, j]
                        + sigma_x * u_star[i - 1, j]
                    )

            u[i, :, k + 1] = thomas_algorithm(A, B, C, D)

        # Устанавливаем граничные значения по x для нового временного слоя
        u[0, :, k + 1] = np.cosh(y) * exp_next
        u[I, :, k + 1] = 0.0

    computation_time = time.time() - start_time
    print(f"Время вычислений: {computation_time:.3f} сек")

    return x, y, t, u


def solve_FS(I, J, K):
    print("\nМЕТОД ДРОБНЫХ ШАГОВ")
    print(f"Сетка: {I}×{J} точек, {K} шагов по времени")

    hx = Lx / I
    hy = Ly / J
    tau = T_final / K

    x = np.linspace(0, Lx, I + 1)
    y = np.linspace(0, Ly, J + 1)
    t = np.linspace(0, T_final, K + 1)

    u = np.zeros((I + 1, J + 1, K + 1))

    # Начальное условие
    for i in range(I + 1):
        for j in range(J + 1):
            u[i, j, 0] = np.cos(2 * x[i]) * np.cosh(y[j])

    # Установим граничные условия первого рода на всех временных слоях
    for k in range(K + 1):
        tk = t[k]
        exp_factor = np.exp(-3 * a * tk)
        u[0, :, k] = np.cosh(y) * exp_factor  # x = 0
        u[I, :, k] = 0.0  # x = pi/4
        u[:, 0, k] = np.cos(2 * x) * exp_factor  # y = 0

    sigma_x = a * tau / hx**2
    sigma_y = a * tau / hy**2

    start_time = time.time()

    for k in range(K):
        t_next = t[k + 1]
        exp_next = np.exp(-3 * a * t_next)

        u_star = np.zeros((I + 1, J + 1))

        # === ПЕРВЫЙ ШАГ: неявно по x (оператор Lx) ===
        for j in range(1, J):  # внутренние узлы по y
            A = np.zeros(I + 1)
            B = np.zeros(I + 1)
            C = np.zeros(I + 1)
            D = np.zeros(I + 1)

            for i in range(I + 1):
                if i == 0:
                    B[i] = 1.0
                    D[i] = np.cosh(y[j]) * exp_next
                elif i == I:
                    B[i] = 1.0
                    D[i] = 0.0
                else:
                    A[i] = -sigma_x
                    B[i] = 1.0 + 2 * sigma_x
                    C[i] = -sigma_x
                    D[i] = u[i, j, k]

            u_star[:, j] = thomas_algorithm(A, B, C, D)

        u_star[0, :] = np.cosh(y) * exp_next
        u_star[I, :] = 0.0  # x = pi/4

        u_star[:, 0] = np.cos(2 * x) * exp_next

        # === ПЕРВЫЙ ШАГ: неявно по y (оператор Ly) ===
        for i in range(1, I):
            A = np.zeros(J + 1)
            B = np.zeros(J + 1)
            C = np.zeros(J + 1)
            D = np.zeros(J + 1)

            for j in range(J + 1):
                if j == 0:
                    B[j] = 1.0
                    D[j] = np.cos(2 * x[i]) * exp_next
                elif j == J:
                    B[j] = 3.0
                    A[j] = -3.0
                    C[j] = 0.0
                    D[j] = -hy * np.cos(2 * x[i]) * exp_next
                else:
                    A[j] = -sigma_y
                    B[j] = 1.0 + 2 * sigma_y
                    C[j] = -sigma_y
                    D[j] = u_star[i, j]  # Явный член: u^*

            u[i, :, k + 1] = thomas_algorithm(A, B, C, D)

        u[0, :, k + 1] = np.cosh(y) * exp_next
        u[I, :, k + 1] = 0.0

    computation_time = time.time() - start_time
    print(f"Время вычислений: {computation_time:.3f} сек")

    return x, y, t, u


def plot_3d_solution_and_error(x, y, t, u_num, method_name):
    import os

    os.makedirs("graphics", exist_ok=True)

    X, Y = np.meshgrid(x, y, indexing="ij")
    time_indices = [len(t) // 3, 2 * len(t) // 3, len(t) - 1]

    for k in time_indices:
        u_exact_k = u_exact(X, Y, t[k])
        u_k = u_num[:, :, k]
        error_k = np.abs(u_k - u_exact_k)

        fig = plt.figure(figsize=(18, 6))

        # ====== ЧИСЛЕННОЕ РЕШЕНИЕ ======
        ax1 = fig.add_subplot(1, 3, 1, projection="3d")
        surf1 = ax1.plot_surface(
            X, Y, u_k, cmap="viridis", linewidth=0, antialiased=True
        )
        ax1.set_title(f"{method_name}: численное решение\n t = {t[k]:.3f}")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("u")
        fig.colorbar(surf1, ax=ax1, shrink=0.5)

        # ====== ТОЧНОЕ РЕШЕНИЕ ======
        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        surf2 = ax2.plot_surface(
            X, Y, u_exact_k, cmap="viridis", linewidth=0, antialiased=True
        )
        ax2.set_title(f"Точное решение\n t = {t[k]:.3f}")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("u")
        fig.colorbar(surf2, ax=ax2, shrink=0.5)

        # ====== ПОГРЕШНОСТЬ ======
        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        surf3 = ax3.plot_surface(
            X, Y, error_k, cmap="hot", linewidth=0, antialiased=True
        )
        ax3.set_title(f"Ошибка |u − u_exact|\n t = {t[k]:.3f}")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.set_zlabel("Error")
        fig.colorbar(surf3, ax=ax3, shrink=0.5)

        plt.tight_layout()

        # ====== СОХРАНЕНИЕ ======
        filename = f"graphics/{method_name}_t_{t[k]:.3f}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Сохранён файл: {filename}")

        plt.show()
        plt.close(fig)


def compute_final_max_error(u_num, x, y, t):
    X, Y = np.meshgrid(x, y, indexing="ij")
    u_exact_T = u_exact(X, Y, t[-1])
    error = np.abs(u_num[:, :, -1] - u_exact_T)
    return np.max(error)


def plot_error_vs_step(
    grid_sizes,
    K,
    method,
):
    hs = []
    errors = []

    for I in grid_sizes:
        J = I
        h = Lx / I
        hs.append(h)

        print(f"\nСетка I = J = {I}, h = {h:.5e}")

        if method == "ADI":
            x, y, t, u = solve_ADI(I, J, K)
        elif method == "FS":
            x, y, t, u = solve_FS(I, J, K)
        else:
            raise ValueError("method должен быть 'ADI' или 'FS'")

        err = compute_final_max_error(u, x, y, t)
        errors.append(err)

        print(f"Макс. ошибка при t=T: {err:.6e}")

    # ====== ГРАФИК ======
    plt.figure(figsize=(9, 6))
    plt.loglog(hs, errors, "o-", linewidth=2, markersize=6)
    plt.xlabel("Шаг сетки h")
    plt.ylabel("max |u − u_exact| при t = T")
    plt.title(f"Зависимость ошибки от шага сетки ({method})")
    plt.grid(True, which="both", alpha=0.3)

    # ====== ПОДПИСИ ЗНАЧЕНИЙ ======
    for h, err in zip(hs, errors):
        plt.annotate(
            f"{err:.2e}",
            xy=(h, err),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            color="black",
        )

    plt.savefig(f"graphics/error_vs_h_{method}.png", dpi=300, bbox_inches="tight")
    plt.show()

    return np.array(hs), np.array(errors)


def main():
    I = 40
    J = 40
    K = 1000

    print("\nПараметры задачи:")
    print(f"  Область: x ∈ [0, {Lx:.4f}], y ∈ [0, {Ly:.4f}], t ∈ [0, {T_final}]")
    print(f"  Коэффициент теплопроводности: a = {a}")
    print("\nПараметры сетки:")
    print(f"  I = {I} (точек по x: {I + 1})")
    print(f"  J = {J} (точек по y: {J + 1})")
    print(f"  K = {K} (шагов по времени)")
    print(f"  hx = {Lx / I:.6f}, hy = {Ly / J:.6f}, τ = {T_final / K:.6f}")

    # 1. РЕШЕНИЕ МЕТОДОМ ПЕРЕМЕННЫХ НАПРАВЛЕНИЙ
    x, y, t, u_ADI = solve_ADI(I, J, K)

    # 2. РЕШЕНИЕ МЕТОДОМ ДРОБНЫХ ШАГОВ
    x, y, t, u_FS = solve_FS(I, J, K)

    # 3. ГРАФИКИ
    plot_3d_solution_and_error(x, y, t, u_ADI, method_name="ADI")
    plot_3d_solution_and_error(x, y, t, u_FS, method_name="Fractional Steps")

    # 4. ИССЛЕДОВАНИЕ СХОДИМОСТИ
    grid_sizes = [10, 20, 40, 80]
    plot_error_vs_step(grid_sizes=grid_sizes, K=1000, method="ADI")
    plot_error_vs_step(grid_sizes=grid_sizes, K=1000, method="FS")


# ================= ЗАПУСК ПРОГРАММЫ =================
if __name__ == "__main__":
    plt.rcParams["figure.figsize"] = [10, 6]
    plt.rcParams["font.size"] = 10

    main()
