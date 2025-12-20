#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <cstring>

#include <cuda_runtime.h>

#define INF 1e30f

const int BLOCK_SIZE = 32;

struct Vec3 {
    float x, y, z;

    __host__ __device__ Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ Vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__ Vec3 operator+(const Vec3& b) const {
        return Vec3(x+b.x, y+b.y, z+b.z);
    }
    __host__ __device__ Vec3 operator-(const Vec3& b) const {
        return Vec3(x-b.x, y-b.y, z-b.z);
    }
    __host__ __device__ Vec3 operator*(float k) const {
        return Vec3(x*k, y*k, z*k);
    }
    __host__ __device__ Vec3 operator*(const Vec3& b) const {
        return Vec3(x*b.x, y*b.y, z*b.z);
    }
};

__host__ __device__ inline float dot(const Vec3& a, const Vec3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__host__ __device__ inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    );
}

__host__ __device__ inline Vec3 normalize(const Vec3& v) {
    float l = sqrtf(dot(v,v));
    if (l < 1e-8f) return Vec3(0,0,0);
    return Vec3(v.x/l, v.y/l, v.z/l);
}

struct Ray {
    Vec3 o, d;
};

struct CameraParams {
    float rc0, zc0, phic0;
    float Acr, Acz, wcr, wcz, wcphi, pcr, pcz;

    float rn0, zn0, phin0;
    float Anr, Anz, wnr, wnz, wnphi, pnr, pnz;
};

__host__ __device__
Vec3 cyl_to_cart(float r, float phi, float z) {
    return Vec3(r*cosf(phi), r*sinf(phi), z);
}

__host__ __device__
Ray make_ray(int x, int y, int w, int h, float fov, float t, const CameraParams& c) {
    // Позиция камеры
    float rc = c.rc0 + c.Acr * sinf(c.wcr * t + c.pcr);
    float zc = c.zc0 + c.Acz * sinf(c.wcz * t + c.pcz);
    float phic = c.phic0 + c.wcphi * t;

    // Точка направления
    float rn = c.rn0 + c.Anr * sinf(c.wnr * t + c.pnr);
    float zn = c.zn0 + c.Anz * sinf(c.wnz * t + c.pnz);
    float phin = c.phin0 + c.wnphi * t;

    Vec3 pos = cyl_to_cart(rc, phic, zc);
    Vec3 look = cyl_to_cart(rn, phin, zn);

    Vec3 forward = normalize(look - pos);
    Vec3 right = normalize(cross(forward, Vec3(0,0,1)));
    Vec3 up = cross(right, forward);

    float aspect = (float)w / h;
    float tanfov = tanf(fov * 0.5f);
    float px = (2.0f * (x + 0.5f) / w - 1.0f) * tanfov * aspect;
    float py = (1.0f - 2.0f * (y + 0.5f) / h) * tanfov;

    Vec3 dir = normalize(forward + right * px + up * py);
    return {pos, dir};
}

struct Triangle {
    Vec3 a, b, c;
    Vec3 color;
};

__host__ __device__
bool intersect_triangle(const Ray& ray, const Triangle& t, float& dist, Vec3& normal) {
    Vec3 e1 = t.b - t.a;
    Vec3 e2 = t.c - t.a;
    Vec3 pvec = cross(ray.d, e2);
    float det = dot(e1, pvec);

    if (fabsf(det) < 1e-8f) return false;

    float inv_det = 1.0f / det;
    Vec3 tvec = ray.o - t.a;
    float u = dot(tvec, pvec) * inv_det;
    if (u < 0.0f || u > 1.0f) return false;

    Vec3 qvec = cross(tvec, e1);
    float v = dot(ray.d, qvec) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return false;

    float tt = dot(e2, qvec) * inv_det;
    if (tt < 1e-4f) return false;

    dist = tt;
    normal = normalize(cross(e1, e2));
    return true;
}

__host__ __device__
Vec3 shade(const Vec3& hit_point, const Vec3& normal, const Vec3& color, const Vec3& light_pos) {
    // Ambient
    float ambient = 0.2f;

    // Diffuse
    Vec3 light_dir = normalize(light_pos - hit_point);
    float diff = fmaxf(0.0f, dot(normal, light_dir));

    Vec3 result = color * (ambient + diff * 0.8f);

    result.x = fminf(1.0f, result.x);
    result.y = fminf(1.0f, result.y);
    result.z = fminf(1.0f, result.z);

    return result;
}

__global__
void render_gpu(Vec3* fb, int w, int h, float fov, float t,
                CameraParams cam, Triangle* tris, int ntris, Vec3 light) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    Ray ray = make_ray(x, y, w, h, fov, t, cam);

    float best_dist = INF;
    Vec3 best_normal, best_color;
    bool hit = false;

    for (int i = 0; i < ntris; i++) {
        float dist;
        Vec3 normal;
        if (intersect_triangle(ray, tris[i], dist, normal) && dist < best_dist) {
            best_dist = dist;
            best_normal = normal;
            best_color = tris[i].color;
            hit = true;
        }
    }

    if (hit) {
        Vec3 hit_point = ray.o + ray.d * best_dist;
        fb[y * w + x] = shade(hit_point, best_normal, best_color, light);
    } else {
        fb[y * w + x] = Vec3(0, 0, 0);
    }
}

void render_cpu(std::vector<Vec3>& fb, int w, int h, float fov, float t,
                const CameraParams& cam, const std::vector<Triangle>& tris, Vec3 light) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            Ray ray = make_ray(x, y, w, h, fov, t, cam);

            float best_dist = INF;
            Vec3 best_normal, best_color;
            bool hit = false;

            for (const auto& tri : tris) {
                float dist;
                Vec3 normal;
                if (intersect_triangle(ray, tri, dist, normal) && dist < best_dist) {
                    best_dist = dist;
                    best_normal = normal;
                    best_color = tri.color;
                    hit = true;
                }
            }

            if (hit) {
                Vec3 hit_point = ray.o + ray.d * best_dist;
                fb[y * w + x] = shade(hit_point, best_normal, best_color, light);
            } else {
                fb[y * w + x] = Vec3(0, 0, 0);
            }
        }
    }
}

void add_tetrahedron(std::vector<Triangle>& tris, Vec3 center, float radius, Vec3 color) {
    float a = radius * 2.0f / sqrtf(6.0f);
    Vec3 v[4] = {
        center + Vec3(0, 0, a * sqrtf(6.0f) / 2.0f),
        center + Vec3(-a * sqrtf(3.0f) / 2.0f, -a / 2.0f, 0),
        center + Vec3(a * sqrtf(3.0f) / 2.0f, -a / 2.0f, 0),
        center + Vec3(0, a, 0)
    };

    tris.push_back({v[0], v[1], v[2], color});
    tris.push_back({v[0], v[2], v[3], color});
    tris.push_back({v[0], v[3], v[1], color});
    tris.push_back({v[1], v[3], v[2], color});
}

void add_octahedron(std::vector<Triangle>& tris, Vec3 center, float radius, Vec3 color) {
    Vec3 v[6] = {
        center + Vec3(radius, 0, 0),
        center + Vec3(-radius, 0, 0),
        center + Vec3(0, radius, 0),
        center + Vec3(0, -radius, 0),
        center + Vec3(0, 0, radius),
        center + Vec3(0, 0, -radius)
    };

    tris.push_back({v[0], v[2], v[4], color});
    tris.push_back({v[2], v[1], v[4], color});
    tris.push_back({v[1], v[3], v[4], color});
    tris.push_back({v[3], v[0], v[4], color});
    tris.push_back({v[0], v[5], v[2], color});
    tris.push_back({v[2], v[5], v[1], color});
    tris.push_back({v[1], v[5], v[3], color});
    tris.push_back({v[3], v[5], v[0], color});
}

void add_dodecahedron(std::vector<Triangle>& tris, Vec3 center, float radius, Vec3 color) {
    const float phi = (1.0f + sqrtf(5.0f)) / 2.0f;
    float scale = radius / sqrtf(3.0f);

    Vec3 v[20] = {
        center + Vec3(1, 1, 1) * scale,
        center + Vec3(1, 1, -1) * scale,
        center + Vec3(1, -1, 1) * scale,
        center + Vec3(1, -1, -1) * scale,
        center + Vec3(-1, 1, 1) * scale,
        center + Vec3(-1, 1, -1) * scale,
        center + Vec3(-1, -1, 1) * scale,
        center + Vec3(-1, -1, -1) * scale,

        center + Vec3(0, phi, 1.0f/phi) * scale,
        center + Vec3(0, phi, -1.0f/phi) * scale,
        center + Vec3(0, -phi, 1.0f/phi) * scale,
        center + Vec3(0, -phi, -1.0f/phi) * scale,

        center + Vec3(1.0f/phi, 0, phi) * scale,
        center + Vec3(-1.0f/phi, 0, phi) * scale,
        center + Vec3(1.0f/phi, 0, -phi) * scale,
        center + Vec3(-1.0f/phi, 0, -phi) * scale,

        center + Vec3(phi, 1.0f/phi, 0) * scale,
        center + Vec3(phi, -1.0f/phi, 0) * scale,
        center + Vec3(-phi, 1.0f/phi, 0) * scale,
        center + Vec3(-phi, -1.0f/phi, 0) * scale
    };

    // 12 пятиугольных граней, каждая разбита на 3 треугольника
    int faces[12][5] = {
        {0, 8, 4, 13, 12},
        {0, 12, 2, 17, 16},
        {0, 16, 1, 9, 8},
        {1, 14, 3, 17, 16},
        {1, 9, 5, 15, 14},
        {2, 10, 6, 13, 12},
        {2, 17, 3, 11, 10},
        {3, 11, 7, 15, 14},
        {4, 8, 9, 5, 18},
        {4, 18, 19, 6, 13},
        {5, 18, 19, 7, 15},
        {6, 19, 7, 11, 10}
    };

    for (int i = 0; i < 12; i++) {
        Vec3 center_face = (v[faces[i][0]] + v[faces[i][1]] + v[faces[i][2]] +
                           v[faces[i][3]] + v[faces[i][4]]) * 0.2f;
        for (int j = 0; j < 5; j++) {
            tris.push_back({center_face, v[faces[i][j]], v[faces[i][(j+1)%5]], color});
        }
    }
}

void add_floor(std::vector<Triangle>& tris, Vec3 p1, Vec3 p2, Vec3 p3, Vec3 p4, Vec3 color) {
    tris.push_back({p1, p2, p3, color});
    tris.push_back({p1, p3, p4, color});
}

int main(int argc, char** argv) {
    // Обработка --default
    if (argc > 1 && strcmp(argv[1], "--default") == 0) {
        printf("60\n");
        printf("./out/img_%%d.data\n");
        printf("640 480 90\n");
        printf("5.0 2.0 0.0 0.0 0.0 1.0 1.0 0.5 0.0 0.0\n");
        printf("0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.0 0.0\n");
        printf("0.0 0.0 0.5 1.0 0.0 0.0 1.0 0.0 0.0 0\n");
        printf("2.0 0.0 0.5 0.0 1.0 0.0 1.0 0.0 0.0 0\n");
        printf("-2.0 0.0 0.5 0.0 0.0 1.0 1.0 0.0 0.0 0\n");
        printf("-5.0 -5.0 -1.0 -5.0 5.0 -1.0 5.0 5.0 -1.0 5.0 -5.0 -1.0 dummy_texture.data 0.5 0.5 0.5 0.0\n");
        printf("1\n");
        printf("5.0 5.0 5.0 1.0 1.0 1.0\n");
        printf("1 1\n");
        return 0;
    }

    bool use_gpu = true;
    if (argc > 1 && strcmp(argv[1], "--cpu") == 0) use_gpu = false;

    // Чтение параметров
    int frames;
    scanf("%d", &frames);

    char path[256];
    scanf("%s", path);

    int W, H;
    float fov_deg;
    scanf("%d%d%f", &W, &H, &fov_deg);
    float fov = fov_deg * M_PI / 180.0f;

    CameraParams cam;
    scanf("%f%f%f%f%f%f%f%f%f%f",
        &cam.rc0, &cam.zc0, &cam.phic0, &cam.Acr, &cam.Acz,
        &cam.wcr, &cam.wcz, &cam.wcphi, &cam.pcr, &cam.pcz);
    scanf("%f%f%f%f%f%f%f%f%f%f",
        &cam.rn0, &cam.zn0, &cam.phin0, &cam.Anr, &cam.Anz,
        &cam.wnr, &cam.wnz, &cam.wnphi, &cam.pnr, &cam.pnz);

    // Чтение параметров тел
    std::vector<Triangle> tris;

    for (int i = 0; i < 3; i++) {
        Vec3 center, color;
        float radius, kr, kt;
        int lights_on_edge;
        scanf("%f%f%f%f%f%f%f%f%f%d",
              &center.x, &center.y, &center.z,
              &color.x, &color.y, &color.z,
              &radius, &kr, &kt, &lights_on_edge);

        // Вариант 4: Тетраэдр, Октаэдр, Додекаэдр
        if (i == 0) {
            add_tetrahedron(tris, center, radius, color);
        } else if (i == 1) {
            add_octahedron(tris, center, radius, color);
        } else if (i == 2) {
            add_dodecahedron(tris, center, radius, color);
        }
    }

    // Чтение параметров пола
    Vec3 floor_p[4];
    for (int i = 0; i < 4; i++) {
        scanf("%f%f%f", &floor_p[i].x, &floor_p[i].y, &floor_p[i].z);
    }
    char tex_path[256];
    scanf("%s", tex_path);
    Vec3 floor_color;
    float floor_refl;
    scanf("%f%f%f%f", &floor_color.x, &floor_color.y, &floor_color.z, &floor_refl);

    add_floor(tris, floor_p[0], floor_p[1], floor_p[2], floor_p[3], floor_color);

    // Чтение источников света
    int num_lights;
    scanf("%d", &num_lights);

    Vec3 light;
    scanf("%f%f%f", &light.x, &light.y, &light.z);

    for (int i = 0; i < num_lights; i++) {
        float dummy;
        scanf("%f%f%f", &dummy, &dummy, &dummy);
    }

    int max_depth, ssaa;
    scanf("%d%d", &max_depth, &ssaa);

    std::vector<Vec3> fb(W * H);

    Vec3* d_fb = nullptr;
    Triangle* d_tris = nullptr;

    if (use_gpu) {
        cudaMalloc(&d_fb, W * H * sizeof(Vec3));
        cudaMalloc(&d_tris, tris.size() * sizeof(Triangle));
        cudaMemcpy(d_tris, tris.data(), tris.size() * sizeof(Triangle), cudaMemcpyHostToDevice);
    }

    // Рендеринг кадров
    double total_time = 0.0;
    for (int f = 0; f < frames; f++) {
        float t = 2.0f * M_PI * f / frames;

        auto start = std::chrono::high_resolution_clock::now();

        if (use_gpu) {
            dim3 block(BLOCK_SIZE, BLOCK_SIZE);
            dim3 grid((W + BLOCK_SIZE - 1) / BLOCK_SIZE, (H + BLOCK_SIZE - 1) / BLOCK_SIZE);
            render_gpu<<<grid, block>>>(d_fb, W, H, fov, t, cam, d_tris, tris.size(), light);
            cudaMemcpy(fb.data(), d_fb, W * H * sizeof(Vec3), cudaMemcpyDeviceToHost);
        } else {
            render_cpu(fb, W, H, fov, t, cam, tris, light);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        total_time += ms;

        // Запись в файл
        char fname[512];
        sprintf(fname, path, f);
        FILE* out = fopen(fname, "wb");
        if (!out) {
            fprintf(stderr, "ERROR: Cannot open file %s\n", fname);
            return 0;
        }
        fwrite(fb.data(), sizeof(Vec3), W * H, out);
        fclose(out);

        // Статистика
        printf("%d\t%.3f\t%d\n", f, ms, W * H);
    }
    printf("TOTAL TIME: %.3f ms\n", total_time);
    printf("AVG TIME: %.3f ms\n", total_time / frames);

    if (use_gpu) {
        cudaFree(d_fb);
        cudaFree(d_tris);
    }

    return 0;
}
