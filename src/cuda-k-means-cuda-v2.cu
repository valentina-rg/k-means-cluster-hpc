// Autore: Sergio Dobrianskiy 1019553
// Versione "v2" ottimizzata della soluzione CUDA

/****************************************************************************
 *
 * k-means.c -- Implementation of the K-Means clustering algorithm.
 *
 * Copyright (C) 2025 Moreno Marzolla <https://unibo.it/sitoweb/moreno.marzolla/>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * --------------------------------------------------------------------------
 *
 * Compile with:
 *
 *      gcc -std=c99 -Wall -Wpedantic k-means.c -o k-means
 *
 * Run with:
 *
 *      ./k-means K input_file output_file
 *
 * Example:
 *
 *      ./k-means 5 demo.in demo.out
 *
 * To generate a movie, you need gnuplot and ffmpeg (the following
 * commands will generate a lot of temporary files):
 *
 *      # compile enabling generation of intermediate results
 *      gcc -DMAKE_MOVIE -std=c99 -Wall -Wpedantic k-means.c -o k-means
 *      # run the program
 *      ./k-means 5 demo.in demo.out
 *      # generate frames
 *      ./generate_frames.sh
 *      # assemble the frames to produce the file "demo.mp4"
 *      ffmpeg -pattern_type glob -stream_loop 5 -y -r 1 -i "img_*.png" -vcodec mpeg4 -r 1 demo.mp4
 *
 ****************************************************************************/

/* Enable POSIX extensions that are required for using
   `clock_gettime()` (used in "hpc.h"). The following `#define` must
   come at the very beginning, before including anything else. */
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include <assert.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "hpc.h"

int h_dims; /* number of dimensions.                */

int n_points; /* number of data points.               */

int n_clusters; /* number of clusters.                  */

float* data; /* [array of length (n_points * n_dims)]
                `&data[i*n_dims]` points to the beginning
                of the i-th data items, which is an array
                of `n_dims` floating-point numbers.  */

float* centroids; /* [array of length (n_clusters * n_dims)]
                     `&centroids[j*n_dims]` points to the
                     beginning of the j-th centroid, which is an
                     array of `n_dims` floating point
                     numbers.                             */

float* new_centroids; /* [array of length (n_clusters * n_dims)] */

int* counts; /* [array of length n_clusters] `counts[j]`
                is the number of points that belong to
                cluster j.                           */

int* cluster_of; /* [array of length n_points] `clusters_of[i]`
                    is the ID of the cluster assigned to the
                    i-th data point; cluster IDs are integer in
                    0..(n_clusters-1).                   */

__host__ __device__ inline int IDX(int row, int col, int stride);

/* A safe version of `malloc()` that aborts if memory allocation
   fails. */
void* safe_malloc(size_t size) {
    void* result = malloc(size);
    assert(result != NULL);
    return result;
}

/******************************************************************************
 **
 ** Utility functions that operate on arrays of `n_dims` elements.
 **
 ******************************************************************************/

#define BLOCK_SIZE 256

/* Set all components of vector `p` of size `n_dims` equal to zero. */
__host__ __device__ inline void vzero(float* p, int n_dims) {
    for (int d = 0; d < n_dims; d++)
        p[d] = 0.0f;
}

/* Add vector `p1` to vector `p2`; store result in `p1`. Both vectors
   have size `n_dims`. */
__host__ __device__ inline void vadd(float* p1, const float* p2, int n_dims) {
    for (int d = 0; d < n_dims; d++)
        p1[d] += p2[d];
}

/* Multiply each element of vector `p` of size `n_dims` by `v`. */
__host__ __device__ inline void vmul(float* p, float v, int n_dims) {
    for (int d = 0; d < n_dims; d++)
        p[d] *= v;
}

/* Copy `p2` into `p1`. */
__host__ __device__ inline void vcopy(float* dest, const float* source, int n_dims) {
    for (int d = 0; d < n_dims; d++)
        dest[d] = source[d];
}

/* Compute the Euclidean squared distance of `p1` and `p2`. */
__host__ __device__ inline float sqdist(const float* p1, const float* p2, int n_dims) {
    float result = 0.0;
    for (int d = 0; d < n_dims; d++) {
        result += (p1[d] - p2[d]) * (p1[d] - p2[d]);
    }
    return result;
}

/* ================================================================
   Kernel ottimizzato – centroidi caricati in shared memory
   ================================================================ */
// __restrict__ = Il programmatore promette al compilatore che nessun altro puntatore
// (tra quelli qualificati con __restrict__) punterà alla stessa zona
// di memoria durante l'esecuzione del kernel.
__global__ void kernel_kmeans_assign_accum_opt(const float* __restrict__ d_points,
                                               const float* __restrict__ d_centroids,
                                               float* __restrict__ d_new_sums,
                                               int* __restrict__ d_counts,
                                               int* __restrict__ d_assignment,
                                               int N,
                                               int K,
                                               int D) {

    // __shared__ = la variabile risiede nella shared memory del thread block.
    // extern = memoria non è definita localmente nel codice 
    // (non ha una dimensione fissa a compile time)
    // CUDA la alloca per block al momento del lancio
    extern __shared__ float s_centroids[];  // spazio dinamico per K*D float

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Fase 1: tutti i thread del blocco caricano i centroidi in shared memory
    for (int idx = threadIdx.x; idx < K * D; idx += blockDim.x) {
        s_centroids[idx] = d_centroids[idx];
    }
    __syncthreads();

    if (i >= N) 
        return;

    const float* point = &d_points[IDX(i, 0, D)];

    int nearest = 0;
    float min_dist = sqdist(point, &s_centroids[IDX(0, 0, D)], D);

    for (int c = 1; c < K; c++) {
        float dist = sqdist(point, &s_centroids[IDX(c, 0, D)], D);
        if (dist < min_dist) {
            min_dist = dist;
            nearest = c;
        }
    }

    d_assignment[i] = nearest;

    atomicAdd(&d_counts[nearest], 1);

    float* target = &d_new_sums[IDX(nearest, 0, D)];
    for (int d = 0; d < D; d++) {
        atomicAdd(&target[d], point[d]);
    }
}

/* ================================================================
   Kernel 2 – normalizzazione centroidi + calcolo max shift
   (stesso di prima, non cambia molto)
   ================================================================ */

__global__ void kernel_finalize_centroids(float* __restrict__ d_new_sums,
                                          const float* __restrict__ d_old_centroids,
                                          const int* __restrict__ d_counts,
                                          float* __restrict__ d_max_shift,
                                          int K,
                                          int D) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= K)
        return;

    float shift_sq = 0.0f;
    float* target = &d_new_sums[IDX(c, 0, D)];

    if (d_counts[c] > 0) {
        float inv = 1.0f / (float)d_counts[c];
        for (int dim = 0; dim < D; dim++) {
            float oldv = d_old_centroids[IDX(c, dim, D)];
            float sumv = target[dim];
            float newv = sumv * inv;
            target[dim] = newv;

            float diff = newv - oldv;
            shift_sq += diff * diff;
        }
    } else {
        for (int dim = 0; dim < D; dim++) {
            target[dim] = d_old_centroids[IDX(c, dim, D)];
        }
    }

    atomicMax((int*)d_max_shift, __float_as_int(shift_sq));
}

/******************************************************************************
 **
 ** K-Means algorithm begins here.
 **
 ******************************************************************************/

/* This function can be used to access the arrays (actually, matrices)
   `data`, `centroids` and `new_centroids`. These are all matrices
   with `n_dims` columns. The function returns the linear index of row
   `i` and column `d`. Example: `data[IDX(i, d)]` is equivalent to
   `data[i*n_dims + d]`. */
__host__ __device__ inline int IDX(int row, int col, int stride) {
    return row * stride + col;
}

/* Return a random integer in a..b. This function must not be
   parallelized, since `rand()` is not thread-safe. */
int randab(int a, int b) {
    return a + rand() % (b - a + 1);
}

/* Centroids are initialized by randomly selecting `n_clusters` data
   points. To select `n_clusters` out of `n_data` elements, we use
   Knuths' algorithm as reported in J. Bentley, "Programming Pearls",
   2nd ed., Addison-Wesley, 2000, p. 126.

   DO NOT PARALLELIZE THIS FUNCTION: `rand()` is not thread-safe. */
void init_centroids(void) {  //* NO KERNEL
    int select = n_clusters;
    int remaining = n_points;
    for (int i = 0; (i < n_points) && (select > 0); i++) {
        if ((rand() % remaining) < select) {
            select--;
            /* Select point `i` as one of the centroids. */
            vcopy(&centroids[IDX(select, 0, h_dims)], &data[IDX(i, 0, h_dims)], h_dims);
        }
        remaining--;
    }
}

/******************************************************************************
 **
 ** Input/output functions. DO NOT parallelize them.
 **
 ******************************************************************************/

/* Read the input data from `f`. Each row must contain `n_dims`
   numbers. This function figures out how many numbers are in a row,
   and how many rows there are. Then, it initializes the variables
   `n_dims` and `n_points` accordingly. */
void read_input(FILE* f) {
    const size_t BUFLEN = 1024;
    char buffer[BUFLEN];

    /* Get the first line of the input file, and count how many
       numbers are there. This function is not very robust: if the
       first line is empty, the number of dimensions will be zero; if
       the first line has more than `BUFLEN` characters, the number of
       fields will be computed incorrectly. */
    char* i_dont_care = fgets(buffer, BUFLEN, f);
    (void)i_dont_care; /* Avoid a compiler warning. */
    h_dims = -1;
    char *start, *end = buffer;
    do {
        start = end;
        strtof(start, &end);
        h_dims++;
    } while (end != start);

    assert(h_dims > 0); /* If this assertion fails, then the first
                           line of the input is empty. */

    /* Rewind the file and count how many data items are there. */
    rewind(f);
    int n_items = 0;
    float dummy;
    while (1 == fscanf(f, "%f", &dummy))
        n_items++;

    assert(n_points % h_dims == 0); /* If this assertion fails, then
                                       there is some line of the input
                                       file that has != n_dims
                                       items. */

    n_points = n_items / h_dims;

    data = (float*)safe_malloc(n_points * h_dims * sizeof(*data));

    /* Rewind and read the actual data. */
    rewind(f);
    for (int i = 0; i < n_points; i++) {
        for (int d = 0; d < h_dims; d++) {
            const int nread = fscanf(f, "%f", &data[IDX(i, d, h_dims)]);
            assert(nread == 1);
        }
    }
}

#ifdef MAKE_MOVIE

/* Save the intermediate coordinates of the centroids into a
   file. This function is useful for generating a movie showing how
   the centroids get updated. DO NOT use this function when measuring
   execution times.

   This function is enabled by defining `MAKE_MOVIE` at compilation
   time. */
void save_centroids(int iter) {
    char buf[1024];

    snprintf(buf, sizeof(buf), "centroids_%03u.txt", (unsigned)iter);
    FILE* f = fopen(buf, "w");
    assert(f != NULL);
    if (f == NULL) {
        fprintf(stderr, "FATAL: can not open file \"%s\" for writing\n", buf);
        exit(EXIT_FAILURE);
    }
    for (int j = 0; j < n_clusters; j++) {
        for (int d = 0; d < h_dims; d++) {
            fprintf(f, "%f ", centroids[IDX(j, d, h_dims)]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

/* Save the intermediate coordinates of the points and their clusters
   into a file. This function is useful for generating a movie showing
   how the centroids get updated, otherwise it can be omitted. DO NOT
   use this function when measuring execution times.

   This function is enabled by defining `MAKE_MOVIE` at compilation
   time.
*/
void save_clusters(int iter) {
    char buf[1024];

    snprintf(buf, sizeof(buf), "clusters_%03u.txt", (unsigned)iter);
    FILE* f = fopen(buf, "w");
    if (f == NULL) {
        fprintf(stderr, "FATAL: can not open file \"%s\" for writing\n", buf);
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < n_points; i++) {
        for (int d = 0; d < h_dims; d++) {
            fprintf(f, "%f ", data[IDX(i, d, h_dims)]);
        }
        fprintf(f, "%d\n", cluster_of[i]);
    }
    fclose(f);
}

#endif

/* Print the final result of the computation, i.e, the coordinates of
   the centroids and the list of data points with the cluster id. */
void save_results(FILE* f) {
    fprintf(f, "# Centroids:\n#\n");
    for (int j = 0; j < n_clusters; j++) {
        fprintf(f, "# %3d :", j);
        for (int d = 0; d < h_dims; d++) {
            fprintf(f, " %f", centroids[IDX(j, d, h_dims)]);
        }
        fprintf(f, "\n");
    }
    fprintf(f, "#\n");
    for (int i = 0; i < n_points; i++) {
        for (int d = 0; d < h_dims; d++) {
            fprintf(f, "%f ", data[IDX(i, d, h_dims)]);
        }
        fprintf(f, "%d\n", cluster_of[i]);
    }
}

/******************************************************************************
 **
 ** Main program.
 **
 ******************************************************************************/
int main(int argc, char* argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s K input_file output_file\n", argv[0]);
        return EXIT_FAILURE;
    }

    srand(123); /* Deterministic initialization of the PRNG. */

    n_clusters = atoi(argv[1]);

    FILE* inputf;
    if ((inputf = fopen(argv[2], "r")) == NULL) {
        fprintf(stderr, "FATAL: can not open input file \"%s\"\n", argv[2]);
        return EXIT_FAILURE;
    }

    read_input(inputf);
    fclose(inputf);

    assert(n_clusters < n_points);

    FILE* outputf;
    if ((outputf = fopen(argv[3], "w")) == NULL) {
        fprintf(stderr, "FATAL: can not create output file \"%s\"\n", argv[3]);
        return EXIT_FAILURE;
    }

    fprintf(outputf, "# Data points: %d\n", n_points);
    fprintf(outputf, "# Dimensions: %d\n", h_dims);
    fprintf(outputf, "# Clusters: %d\n", n_clusters);

    printf("\nInput file....... %s\n", argv[2]);
    printf("Output file...... %s\n", argv[3]);
    printf("Data points (N).. %d\n", n_points);
    printf("Dimensions (D)... %d\n", h_dims);
    printf("Clusters (K)..... %d\n\n", n_clusters);

    // allocazioni host
    centroids = (float*)safe_malloc(n_clusters * h_dims * sizeof(*centroids));
    new_centroids = (float*)safe_malloc(n_clusters * h_dims * sizeof(*new_centroids));
    cluster_of = (int*)safe_malloc(n_points * sizeof(*cluster_of));
    counts = (int*)safe_malloc(n_clusters * sizeof(*counts));

    init_centroids();

    // allocazioni e copie device
    float* d_data = NULL;
    float* d_centroids = NULL;
    float* d_new_sums = NULL;
    int* d_counts = NULL;
    int* d_cluster = NULL;
    float* d_max_shift = NULL;

    cudaMalloc(&d_data, n_points * h_dims * sizeof(float));
    cudaMalloc(&d_centroids, n_clusters * h_dims * sizeof(float));
    cudaMalloc(&d_new_sums, n_clusters * h_dims * sizeof(float));
    cudaMalloc(&d_counts, n_clusters * sizeof(int));
    cudaMalloc(&d_cluster, n_points * sizeof(int));
    cudaMalloc(&d_max_shift, sizeof(float));

    cudaMemcpy(d_data, data, n_points * h_dims * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, centroids, n_clusters * h_dims * sizeof(float), cudaMemcpyHostToDevice);

    const int grid_assign = (n_points + BLOCK_SIZE - 1) / BLOCK_SIZE;
    printf("grid_assign = %d, threads totali = %d\n", grid_assign, grid_assign * BLOCK_SIZE);
    const int grid_update = (n_clusters + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // dimensione shared memory necessaria: K * D * sizeof(float)
    size_t shared_mem_size = n_clusters * h_dims * sizeof(float);
    printf("Shared memory per block: %.1f KB (max ~48 KB su GTX 1070)\n", shared_mem_size / 1024.0);

    // fine allocazioni device

    printf("Main CUDA loop starts (versione ottimizzata con shared memory)\n\n");

    float maxsqshift = 1.0f;
    int iter = 0;
    const float TOL = 1e-5f;
    const int MAXITER = 100;
    double tstart = hpc_gettime();

    while (iter < MAXITER && maxsqshift > TOL * TOL) {
        cudaMemset(d_counts, 0, n_clusters * sizeof(int));
        cudaMemset(d_new_sums, 0, n_clusters * h_dims * sizeof(float));

        kernel_kmeans_assign_accum_opt<<<grid_assign, BLOCK_SIZE, shared_mem_size>>>(
            d_data, d_centroids, d_new_sums, d_counts, d_cluster, n_points, n_clusters, h_dims);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
            exit(1);
        }

        float zero = 0.0f;
        cudaMemcpy(d_max_shift, &zero, sizeof(float), cudaMemcpyHostToDevice);

        kernel_finalize_centroids<<<grid_update, BLOCK_SIZE>>>(d_new_sums, d_centroids, d_counts,
                                                               d_max_shift, n_clusters, h_dims);

        cudaDeviceSynchronize();

        float h_max_shift;
        cudaMemcpy(&h_max_shift, d_max_shift, sizeof(float), cudaMemcpyDeviceToHost);
        maxsqshift = h_max_shift;

        // swap
        float* tmp = d_centroids;
        d_centroids = d_new_sums;
        d_new_sums = tmp;

        printf("Iter %3d   maxsqshift = %.6f\n", iter, maxsqshift);
        iter++;
    }

    double elapsed = hpc_gettime() - tstart;
    printf("\nElapsed time: %.3f s\n", elapsed);

    // copia risultati su device
    cudaMemcpy(centroids, d_centroids, n_clusters * h_dims * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(cluster_of, d_cluster, n_points * sizeof(int), cudaMemcpyDeviceToHost);

    fprintf(outputf, "# Data points: %d\n", n_points);
    fprintf(outputf, "# Dimensions: %d\n", h_dims);
    fprintf(outputf, "# Clusters: %d\n", n_clusters);
    fprintf(outputf, "#\n");

    save_results(outputf);

    fclose(outputf);

    // pulizia host
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_new_sums);
    cudaFree(d_counts);
    cudaFree(d_cluster);
    cudaFree(d_max_shift);

    // pulizia device
    free(data);
    free(centroids);
    free(cluster_of);
    free(counts);

    return EXIT_SUCCESS;
}