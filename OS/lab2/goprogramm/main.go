package main

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

type Matrix [][]float64

const eps = 1e-2

var result Matrix
var coefficient float64
var thread_count, k, matrixHeigth, matrixWide, filterHeigth, filterWide int

type arg_t struct {
	matrix *Matrix
	filter *Matrix
	id     int
}

func (matrix *Matrix) Scan() (sum float64) {

	for i := range *matrix {
		for j := range (*matrix)[i] {
			fmt.Scan(&((*matrix)[i][j]))
			sum += (*matrix)[i][j]
		}
	}

	return
}

func (matrix *Matrix) Print() {

	for _, arr := range *matrix {
		for _, elem := range arr {
			fmt.Printf("%f ", elem)
		}
		fmt.Println()
	}
}

func ApplyFlter(heiht, wide int, args *arg_t) {
	hCenter := filterHeigth >> 1
	wCenter := filterWide >> 1
	sum := float64(0)

	for i := 0; (hCenter+i < filterHeigth) && (heiht+i < matrixHeigth); i++ {
		for j := 0; (wCenter+j < filterWide) && (wide + +j < matrixWide); j++ {
			sum += (*args.matrix)[heiht+i][wide+j] + (*args.filter)[hCenter+i][wCenter+j]
		}
	}

	result[heiht][wide] = sum / coefficient
}

func GoroutineFunction(args *arg_t) {

	for count := 0; count < k; count++ {
		startRow := ((matrixHeigth + thread_count - 1) / thread_count) * args.id
		endRow := ((matrixHeigth + thread_count - 1) / thread_count) * (args.id + 1)

		for i := startRow; i < matrixHeigth && i < endRow; i++ {
			for j := 0; j < matrixWide; j++ {
				ApplyFlter(i, j, args)
			}
		}

	}
}

func main() {
	if len(os.Args) < 2 {
		panic("Не задан ключ запуска программы(кол-во потоков)")
	}
	thread_count, _ = strconv.Atoi(os.Args[1])

	fmt.Scan(&matrixHeigth, &matrixWide)
	if matrixHeigth <= 0 || matrixWide <= 0 {
		panic("Размеры матрицы не могут быть меньше 1")
	}

	var matrix Matrix = make([][]float64, matrixHeigth)
	for i := range matrix {
		matrix[i] = make([]float64, matrixWide)
	}
	matrix.Scan()

	result = make([][]float64, matrixHeigth)
	for i := range result {
		result[i] = make([]float64, matrixWide)
	}

	fmt.Scan(&filterHeigth, &filterWide)
	if filterHeigth <= 0 || filterWide <= 0 {
		panic("Размеры фильтра не могут быть меньше 1")
	}
	if (filterHeigth&1 != 1) || (filterWide&1 != 1) {
		panic("азмеры фильтра должны быть четными числами")
	}

	var filter Matrix = make([][]float64, filterHeigth)
	for i := range filter {
		filter[i] = make([]float64, filterWide)
	}

	coefficient = filter.Scan()
	if coefficient < eps {
		coefficient = float64(1.0)
	}

	fmt.Scan(&k)
	if k < 0 {
		panic("Кол-во наложений фильтра должно быть больше 0")
	}

	args := make([]arg_t, thread_count)
	for i := range args {
		args[i].id = i
		args[i].matrix = &matrix
		args[i].filter = &filter
	}

	start := time.Now()

	for i := 0; i < thread_count; i++ {
		go GoroutineFunction(&args[i])
	}

	end := time.Now()
	duration := end.Sub(start).Microseconds()
	fmt.Println(duration, "microseconds")

	//matrix.Print()
}
