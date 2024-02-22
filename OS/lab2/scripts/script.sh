#!/bin/bash

start_time_mono=$(date +%s.%N)
~/Study/OS/lab2/bin/SyncMain $i < ~/Study/OS/lab2/InOutFiles/1.in > ~/Study/OS/lab2/InOutFiles/1.out
end_time_mono=$(date +%s.%N)
execution_time_mono=$(echo "($end_time_mono - $start_time_mono)*1000" | bc)


for ((i=1; i<=15; i++))
do
    total_time=0

    for ((j=1; j<=10; j++))
    do
        # Запуск ./bin/main и измерение времени выполнения
        start_time=$(date +%s.%N)
        ~/Study/OS/lab2/bin/AssyncMain $i < ~/Study/OS/lab2/InOutFiles/1.in > ~/Study/OS/lab2/InOutFiles/1.out
        end_time=$(date +%s.%N)

        # Вычисление времени выполнения и добавление к общему времени
        execution_time=$(echo "$end_time - $start_time" | bc)
        total_time=$(echo "$total_time + $execution_time" | bc)
    done

    # Вычисление среднего времени выполнения и вывод
    # Вычисление среднего времени выполнения и вывод в миллисекундах
    average_time=$(echo "scale=5; $total_time / 10 * 1000" | bc)
    echo "Среднее время выполнения для $i потоков: $average_time мс"
    (echo $i $average_time) >> ~/Study/OS/lab2/InOutFiles/data.txt
    acceleration=$(echo "scale=5; $execution_time_mono/ $average_time" | bc)
    echo "Ускорение для запуска с $i потоков: 0$acceleration"
    (echo $i $acceleration) >> ~/Study/OS/lab2/InOutFiles/data.txt
    efficient=$(echo "scale=5; $acceleration/$i" | bc)
    echo "Эффективность для запуска с $i потоков: 0$efficient"
    (echo $i $efficient) >> ~/Study/OS/lab2/InOutFiles/data.txt
    echo -e ""
done
