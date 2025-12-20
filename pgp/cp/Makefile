.PHONY: build run_cpu run_gpu run_default clean

clean:
	rm -f out.mp4
	rm -f out/*
	rm -f png/*
	rm -f main

build:
	nvcc main.cu -O2 -std=c++17 -o main

run_cpu: build
	./main --cpu < input.txt
	cd converter && python3 main.py && cd ..
	ffmpeg -framerate 30 -i png/%03d.png out.mp4

run_gpu: build
	./main --gpu < input.txt
	cd converter && python3 main.py && cd ..
	ffmpeg -framerate 30 -i png/%03d.png out.mp4

run_default: build
	./main --default
