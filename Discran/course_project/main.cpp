#define MINIMP3_IMPLEMENTATION
#include "minimp3.h"
#include "minimp3_ex.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <iomanip>
#include <stdexcept>

const int FRAME_SIZE = 4096;
const int OVERLAP = 1024;
const int PRECISION = 8;

std::vector<short> decoder() {
    mp3dec_t mp3d;
    mp3dec_file_info_t info;
    if (mp3dec_load(&mp3d, "input.mp3", &info, NULL, NULL)) {
        throw std::runtime_error("Decode error");
    }

    std::vector<short> samples(info.buffer, info.buffer + info.samples);
    free(info.buffer);
    return samples;
}

void apply_hanning_window(std::vector<short> const& samples, size_t start, std::vector<std::complex<double>>& buffer) {
    size_t N = buffer.size();

    for (size_t idx = 0; idx < N; ++idx) {
        double hann_value = 0.5 * (1 - cos(2 * M_PI * idx / (N - 1)));
        buffer[idx] = std::complex<double>(samples[start+idx] * hann_value, 0.0);
    }
}

void fft(std::vector<std::complex<double>>& buffer) {
    size_t N = buffer.size();
    if (N <= 1) return;

    std::vector<std::complex<double>> even(N / 2);
    std::vector<std::complex<double>> odd(N / 2);
    for (size_t i = 0; i < N / 2; ++i) {
        even[i] = buffer[2 * i];
        odd[i] = buffer[2 * i + 1];
    }

    fft(even);
    fft(odd);

    for (size_t i = 0; i < N / 2; ++i) {
        double angle = 2 * M_PI * i / N;
        auto w = std::complex<double>(cos(angle), sin(angle)) * odd[i];
        buffer[i] = even[i] + w;
        buffer[i + N / 2] = even[i] - w;
    }
}

int main() {
    try {
        auto samples = decoder();

        size_t total_samples = samples.size();
        std::vector<std::complex<double>> fft_buffer(FRAME_SIZE);

        for (size_t start = 0; start + FRAME_SIZE <= total_samples; start += OVERLAP) {
            apply_hanning_window(samples, start, fft_buffer);


            fft(fft_buffer);

            double max_amplitude = std::numeric_limits<double>::min();
            for (const auto& freq_component : fft_buffer) {
                max_amplitude = std::max(max_amplitude, std::abs(freq_component));
            }

            std::cout << std::fixed << std::setprecision(PRECISION) << max_amplitude << '\n';
        }
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }

    return 0;
}
