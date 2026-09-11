/****************************************************************************
 *
 * mpi-k-means.c -- MPI implementation of the K-Means clustering algorithm.
 *
 * Created starting from the serial implementation by Moreno Marzolla <https://unibo.it/sitoweb/moreno.marzolla/>
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

 * Compile with:
 *
 *      mpicc -std=c99 -Wall -Wpedantic mpi-k-means.c -o mpi-k-means
 *
 * Run with:
 *
 *      mpirun -n NPROC ./mpi-k-means K input_file output_file
 *      OR
 *      mpirun --use-hwthread-cpus ./mpi-k-means K input_file output_file
 *
 * Example:
 *
 *      mpirun -n 8 ./mpi-k-means 5 demo.in demo.out
 *      OR
 *      mpirun --use-hwthread-cpus ./mpi-k-means 5 demo.in demo.out
 *
 ****************************************************************************/

/* Enable POSIX extensions that are required for using
   `clock_gettime()` (used in "hpc.h"). The following `#define` must
   come at the very beginning, before including anything else. */
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#define MASTER 0

#include "hpc.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <mpi.h>

int n_dims; /* number of dimensions.                */

int n_points; /* number of data points.               */

int n_clusters; /* number of clusters.                  */

float *data;       /* [array of length (n_points * n_dims)]
                      `&data[i*n_dims]` points to the beginning
                      of the i-th data items, which is an array
                      of `n_dims` floating-point numbers.  */
float *local_data; /* local portion of data */

float *centroids; /* [array of length (n_clusters * n_dims)]
                     `&centroids[j*n_dims]` points to the
                     beginning of the j-th centroid. After the Bcast
                     these are identical on every process (global
                     centroids), not per-process data.     */

float *new_centroids; /* [array of length (n_clusters * n_dims)] */

int *counts; /* [array of length n_clusters] `counts[j]`
                is the number of points that belong to
                cluster j.                           */

int *local_counts; /* Local array of counters */

int *cluster_of; /* [array of length n_points] `clusters_of[i]`
                    is the ID of the cluster assigned to the
                    i-th data point; cluster IDs are integer in
                    0..(n_clusters-1).                   */

int *local_cluster_of;
/* A safe version of `malloc()` that aborts if memory allocation
   fails. */
void *safe_malloc(size_t size)
{
    void *result = malloc(size);
    assert(result != NULL);
    return result;
}

/******************************************************************************
 **
 ** Utility functions that operate on arrays of `n_dims` elements.
 **
 ******************************************************************************/

/* Set all components of vector `p` of size `n_dims` equal to zero. */
void vzero(float *p)
{
    for (int d = 0; d < n_dims; d++)
        p[d] = 0.0f;
}

/* Add vector `p1` to vector `p2`; store result in `p1`. Both vectors
   have size `n_dims`. */
void vadd(float *p1, const float *p2)
{
    for (int d = 0; d < n_dims; d++)
        p1[d] += p2[d];
}

/* Multiply each element of vector `p` of size `n_dims` by `v`. */
void vmul(float *p, float v)
{
    for (int d = 0; d < n_dims; d++)
        p[d] *= v;
}

/* Copy `p2` into `p1`. */
void vcopy(float *p1, const float *p2)
{
    for (int d = 0; d < n_dims; d++)
        p1[d] = p2[d];
}

/* Compute the Euclidean squared distance of `p1` and `p2`. */
float sqdist(float *p1, float *p2)
{
    float result = 0.0;
    for (int d = 0; d < n_dims; d++)
    {
        result += (p1[d] - p2[d]) * (p1[d] - p2[d]);
    }
    return result;
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
int IDX(int i, int d)
{
    return i * n_dims + d;
}

/* Centroids are initialized by randomly selecting `n_clusters` data
   points. To select `n_clusters` out of `n_data` elements, we use
   Knuths' algorithm as reported in J. Bentley, "Programming Pearls",
   2nd ed., Addison-Wesley, 2000, p. 126.

   DO NOT PARALLELIZE THIS FUNCTION: `rand()` is not thread-safe. */
void init_centroids(void)
{
    int select = n_clusters;
    int remaining = n_points;
    for (int i = 0; (i < n_points) && (select > 0); i++)
    {
        if ((rand() % remaining) < select)
        {
            select--;
            /* Select point `i` as one of the centroids. */
            vcopy(&centroids[IDX(select, 0)], &data[IDX(i, 0)]);
        }
        remaining--;
    }
}

/* Assign each data point to the nearest centroid. Updates
   the `counts` array. */
void classify(int my_size, int my_rank)
{
    for (int j = 0; j < n_clusters; j++)
    {
        local_counts[j] = 0;
    }

    for (int i = 0; i < my_size; i++)
    {
        /* Index and squared distance of the nearest centroid. */
        int nearest = 0;
        float mindist = sqdist(&local_data[IDX(i, 0)], &centroids[IDX(nearest, 0)]);
        for (int j = 1; j < n_clusters; j++)
        {
            const float dist = sqdist(&local_data[IDX(i, 0)], &centroids[IDX(j, 0)]);
            if (dist < mindist)
            {
                mindist = dist;
                nearest = j;
            }
        }
        /* assign the point to the nearest centroid, and update the
           cluster size. */
        local_cluster_of[i] = nearest;
        local_counts[nearest]++;
    }
}

/* Update the centroids. Set the centroid of each cluster to the
   barycenter of the points. Returns the maximum shift, i.e., the
   maximum difference between the (squared) old and new position of
   all centroids. */
float update_centroids(int my_size, int my_rank)
{
    for (int j = 0; j < n_clusters; j++)
    {
        vzero(&new_centroids[IDX(j, 0)]);
    }

    for (int i = 0; i < my_size; i++)
    {
        vadd(&new_centroids[IDX(local_cluster_of[i], 0)], &local_data[IDX(i, 0)]);
    }

    /* Computes the global sum of each cluster middle point */
    if (my_rank == MASTER)
    {
        MPI_Reduce(
            MPI_IN_PLACE,        /* send buffer */
            new_centroids,       /* receive buffer */
            n_clusters * n_dims, /* count */
            MPI_FLOAT,           /* data type */
            MPI_SUM,             /* reduce operator */
            MASTER,              /* root */
            MPI_COMM_WORLD);     /* communicator */
    }
    else
    {
        MPI_Reduce(
            new_centroids,       /* send buffer */
            NULL,                /* receive buffer */
            n_clusters * n_dims, /* count */
            MPI_FLOAT,           /* data type */
            MPI_SUM,             /* reduce operator */
            MASTER,              /* root */
            MPI_COMM_WORLD);     /* communicator */
    }

    /**
     * USE OF IN-PLACE OPTION (source "man MPI_Reduce")
     * When  the communicator is an intracommunicator, you can perform a reduce operation in-place (the output buffer is used as the input buffer).  Use
     * the variable MPI_IN_PLACE as the value of the root process sendbuf.  In this case, the input data is taken at the root from the  receive  buffer,
     * where it will be replaced by the output data.
     */

    float maxsqshift = 0.0f;

    /* assuming a low number of cluster, parallelizing is not ideal for performances */
    if (my_rank == MASTER)
    {
        for (int j = 0; j < n_clusters; j++)
        {
            /* If a cluster is empty, we simply copy the old centroid to
               the new one. */
            if (counts[j] == 0)
            {
                vcopy(&new_centroids[IDX(j, 0)], &centroids[IDX(j, 0)]);
            }
            else
            {
                vmul(&new_centroids[IDX(j, 0)], 1.0f / counts[j]);
            }
            const float sqshift = sqdist(&centroids[IDX(j, 0)], &new_centroids[IDX(j, 0)]);
            if (sqshift > maxsqshift)
                maxsqshift = sqshift;
            vcopy(&centroids[IDX(j, 0)], &new_centroids[IDX(j, 0)]);
        }
    }

    /* The master computed maxsqshift; broadcast it so that every process
       evaluates the same loop-termination condition. This collective also
       synchronizes the processes, so no explicit MPI_Barrier is needed. */
    MPI_Bcast(&maxsqshift, 1, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
    return maxsqshift;
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
void read_input(FILE *f)
{
    const size_t BUFLEN = 1024;
    char buffer[BUFLEN];

    /* Get the first line of the input file, and count how many
       numbers are there. This function is not very robust: if the
       first line is empty, the number of dimensions will be zero; if
       the first line has more than `BUFLEN` characters, the number of
       fields will be computed incorrectly. */
    char *i_dont_care = fgets(buffer, BUFLEN, f);
    (void)i_dont_care; /* Avoid a compiler warning. */
    n_dims = -1;
    char *start, *end = buffer;
    do
    {
        start = end;
        strtof(start, &end);
        n_dims++;
    } while (end != start);

    assert(n_dims > 0); /* If this assertion fails, then the first
                           line of the input is empty. */

    /* Rewind the file and count how many data items are there. */
    rewind(f);
    int n_items = 0;
    float dummy;
    while (1 == fscanf(f, "%f", &dummy))
        n_items++;

    assert(n_items % n_dims == 0); /* If this assertion fails, then
                                      there is some line of the input
                                      file that has != n_dims
                                      items. */

    n_points = n_items / n_dims;

    data = (float *)safe_malloc(n_points * n_dims * sizeof(*data));

    /* Rewind and read the actual data. */
    rewind(f);
    for (int i = 0; i < n_points; i++)
    {
        for (int d = 0; d < n_dims; d++)
        {
            const int nread = fscanf(f, "%f", &data[IDX(i, d)]);
            assert(nread == 1);
        }
    }
}

/* Print the final result of the computation, i.e, the coordinates of
   the centroids and the list of data points with the cluster id. */
void save_results(FILE *f)
{
    fprintf(f, "# Centroids:\n#\n");
    for (int j = 0; j < n_clusters; j++)
    {
        fprintf(f, "# %3d :", j);
        for (int d = 0; d < n_dims; d++)
        {
            fprintf(f, " %f", centroids[IDX(j, d)]);
        }
        fprintf(f, "\n");
    }
    fprintf(f, "#\n");
    for (int i = 0; i < n_points; i++)
    {
        for (int d = 0; d < n_dims; d++)
        {
            fprintf(f, "%f ", data[IDX(i, d)]);
        }
        fprintf(f, "%d\n", cluster_of[i]);
    }
}

/******************************************************************************
 **
 ** Main program.
 **
 ******************************************************************************/
int main(int argc, char *argv[])
{
    FILE *inputf, *outputf;
    const int MAXITER = 100;
    const float TOL = 1e-5;

    int my_rank, comm_sz, my_start, my_end, my_size;
    int *displs = NULL, *sendCounts = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

    if (my_rank == MASTER)
    {
        if (argc != 4)
        {
            fprintf(stderr, "Usage: %s K input_file output_file\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, MPI_ERR_IO);
        }

        if ((inputf = fopen(argv[2], "r")) == NULL)
        {
            fprintf(stderr, "FATAL: can not open input file \"%s\"\n", argv[2]);
            MPI_Abort(MPI_COMM_WORLD, MPI_ERR_IO);
        }
    }

    srand(123); /* Deterministic initialization of the PRNG. */
    n_clusters = atoi(argv[1]);

    if (my_rank == MASTER)
    {
        read_input(inputf);
        fclose(inputf);

        assert(n_clusters < n_points);

        if ((outputf = fopen(argv[3], "w")) == NULL)
        {
            fprintf(stderr, "FATAL: can not create output file \"%s\"\n", argv[3]);
            MPI_Abort(MPI_COMM_WORLD, MPI_ERR_IO);
        }

        fprintf(outputf, "# Data points: %d\n", n_points);
        fprintf(outputf, "# Dimensions: %d\n", n_dims);
        fprintf(outputf, "# Clusters: %d\n", n_clusters);

        printf("\nInput file....... %s\n", argv[2]);
        printf("Output file...... %s\n", argv[3]);
        printf("Data points (N).. %d\n", n_points);
        printf("Dimensions (D)... %d\n", n_dims);
        printf("Clusters (K)..... %d\n\n", n_clusters);

        /* Used only by the master */
        cluster_of = (int *)safe_malloc(n_points * sizeof(*cluster_of));
        counts = (int *)safe_malloc(n_clusters * sizeof(*counts));

        /* Data structures needed for the Scatterv and Gatherv primitives */
        displs = (int *)safe_malloc(comm_sz * sizeof(*displs));
        sendCounts = (int *)safe_malloc(comm_sz * sizeof(*sendCounts));
    }

    MPI_Bcast(&n_points,
              1,
              MPI_INT,
              MASTER,
              MPI_COMM_WORLD);

    MPI_Bcast(&n_dims,
              1,
              MPI_INT,
              MASTER,
              MPI_COMM_WORLD);

    MPI_Bcast(&n_clusters,
              1,
              MPI_INT,
              MASTER,
              MPI_COMM_WORLD);

    /* Data needed to generate the displacement and sendCount array */
    my_start = IDX((n_points * my_rank) / comm_sz, 0);
    my_end = IDX((n_points * (my_rank + 1)) / comm_sz, 0);
    my_size = my_end - my_start;
    int my_points = my_size / n_dims;

    local_data = (float *)safe_malloc(my_size * sizeof(*local_data));
    local_cluster_of = (int *)safe_malloc(my_points * sizeof(*local_cluster_of));
    local_counts = (int *)safe_malloc(n_clusters * sizeof(*local_counts));
    centroids = (float *)safe_malloc(n_clusters * n_dims * sizeof(*centroids));
    new_centroids = (float *)safe_malloc(n_clusters * n_dims * sizeof(*new_centroids));

    if (my_rank == MASTER)
    {
        init_centroids();
    }

    // /* Create the displs and sendCount arrays for the scatterv/gatherv */
    // /* Gather displacements */
    MPI_Gather(&my_start,     /* Send buffer */
               1,             /* Item sent count */
               MPI_INT,       /* Item data type */
               displs,        /* Receive buffer */
               1,             /* Item received count */
               MPI_INT,       /* Received item type */
               MASTER,        /* Receiving process rank */
               MPI_COMM_WORLD /* Communicator */
    );

    /* Gather sendCounts */
    MPI_Gather(&my_size,      /* Send buffer */
               1,             /* Item sent count */
               MPI_INT,       /* Item data type */
               sendCounts,    /* Receive buffer */
               1,             /* Item received count */
               MPI_INT,       /* Received item type */
               MASTER,        /* Receiving process rank */
               MPI_COMM_WORLD /* Communicator */
    );

    /* Scatter to each node the points */
    MPI_Scatterv(
        data,
        sendCounts,
        displs,
        MPI_FLOAT,
        local_data,
        my_size,
        MPI_FLOAT,
        MASTER,
        MPI_COMM_WORLD);

    float maxsqshift = 0;
    int iter = 0;
    int clusters_dim = n_clusters * n_dims;

    double tstart = 0.0;

    if (my_rank == MASTER)
        tstart = hpc_gettime();

    MPI_Barrier(MPI_COMM_WORLD);
    do
    {
        /* Sends the current centroids to every process */
        MPI_Bcast(
            centroids,       /* send buffer */
            clusters_dim,    /* count */
            MPI_FLOAT,       /* data type */
            MASTER,          /* root */
            MPI_COMM_WORLD); /* communicator */

        classify(my_points, my_rank); /* Divide by n_dims to get the number of points */

        /* Calculates the number of points in each cluster */
        MPI_Reduce(
            local_counts,  /* send buffer */
            counts,        /* receive buffer */
            n_clusters,    /* number of elements of send buffer */
            MPI_INT,       /* data type */
            MPI_SUM,       /* the reduce operation */
            MASTER,        /* rank of root process */
            MPI_COMM_WORLD /* communicator */
        );

        maxsqshift = update_centroids(my_points, my_rank);
        // printf("Iteration %3d, maxsqshift = %f\n", iter, maxsqshift);
        iter++;
    } while ((maxsqshift > TOL * TOL) && (iter <= MAXITER));

    if (my_rank == MASTER)
    {
        const double elapsed = hpc_gettime() - tstart;
        printf("\nMain loop completed\n");
        printf("Execution time %.3f\n\n", elapsed);

        /* Update displacement and sendcount arrays to gather cluster_of arrays */
        for (int i = 0; i < comm_sz; i++)
        {
            sendCounts[i] /= n_dims;
            displs[i] /= n_dims;
        }
    }

    // To save results, gather cluster_of from all threads.
    MPI_Gatherv(
        local_cluster_of,
        my_points,
        MPI_INT,
        cluster_of,
        sendCounts,
        displs,
        MPI_INT,
        MASTER,
        MPI_COMM_WORLD);

    if (my_rank == MASTER)
    {
        save_results(outputf);
        fclose(outputf);
    }

    free(local_data);
    free(local_cluster_of);
    free(centroids);
    free(local_counts);
    free(new_centroids);

    if (my_rank == MASTER)
    {
        free(data);
        free(cluster_of);
        free(counts);
        free(displs);
        free(sendCounts);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}