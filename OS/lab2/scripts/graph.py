import sys
import matplotlib.pyplot as plt
import os

# Создание списков для хранения чисел
speed_x_values = []
speed_y_values = []
acceleration_x_values = []
acceleration_y_values = []
efficiency_x_values = []
efficiency_y_values = []

# Проверка наличия аргумента командной строки
if len(sys.argv) != 2:
    print('Ошибка: программа должна принимать один аргумент командной строки - путь к файлу')
    sys.exit(1)

# Получение пути к файлу из аргумента командной строки
file_path = sys.argv[1]

# Проверка существования файла
if not os.path.isfile(file_path):
    print('Ошибка: файл не существует')
    sys.exit(1)

# Счетчик для отслеживания количества считанных строк
line_counter = 0

# Считывание строк из файла и получение чисел
with open(file_path, 'r') as file:
    for line_number, line in enumerate(file, 1):
        numbers = line.strip().split()

        # Проверка, что в строке есть два числа
        if len(numbers) != 2:
            print(f'Ошибка в строке {line_number}: в файле должно быть два числа')
            sys.exit(1)

        # Преобразование чисел из строкового формата в числовой
        try:
            x = float(numbers[0])
            y = float(numbers[1])
        except ValueError:
            print(f'Ошибка в строке {line_number}: строки в файле должны содержать числа')
            sys.exit(1)

        # Добавление чисел в соответствующие списки
        if line_counter % 3 == 0:
            speed_x_values.append(x)
            speed_y_values.append(y)
        elif line_counter % 3 == 1:
            acceleration_x_values.append(x)
            acceleration_y_values.append(y)
        elif line_counter % 3 == 2:
            efficiency_x_values.append(x)
            efficiency_y_values.append(y)

        # Увеличение счетчика считанных строк
        line_counter += 1

        # Если считаны уже 15 строк, выходим из цикла
        if line_counter == 45:
            break

# Построение трех графиков после считывания всех строк
# Построение трех графиков
fig, axs = plt.subplots(3)

# График скорости
axs[0].plot(speed_x_values, speed_y_values, '-o')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].set_title('График скорости')

# График ускорения
axs[1].plot(acceleration_x_values, acceleration_y_values, '-o')
axs[1].set_xlabel('X')
axs[1].set_ylabel('Y')
axs[1].set_title('График ускорения')

# График эффективности
axs[2].plot(efficiency_x_values, efficiency_y_values, '-o')
axs[2].set_xlabel('X')
axs[2].set_ylabel('Y')
axs[2].set_title('График эффективности')

# Размещение графиков на холсте
plt.tight_layout()

# Отображение графиков
plt.show()
