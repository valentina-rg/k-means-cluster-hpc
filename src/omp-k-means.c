// Autore: Valentina Rigato
//
//
//

/****************************************************************************
 *
 * omp-k-means.c -- OpenMP implementation of the K-Means clustering algorithm.
 *
 * Created starting from the serial implementation by Moreno Marzolla
 * <https://unibo.it/sitoweb/moreno.marzolla/>
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
 *      gcc -std=c99 -Wall -Wpedantic -fopenmp omp-k-means.c -o omp-k-means
 *
 * Run with:
 *
 *      ./omp-k-means K input_file output_file
 *
 * Example:
 *
 *      ./omp-k-means 5 demo.in demo.out
 *
 * To generate a movie, you need gnuplot and ffmpeg:
 *
 *      gcc -DMAKE_MOVIE -std=c99 -Wall -Wpedantic -fopenmp omp-k-means.c -o omp-k-means
 *      ./omp-k-means 5 demo.in demo.out
 *      ./generate_frames.sh
 *      ffmpeg -pattern_type glob -stream_loop 5 -y -r 1 -i "img_*.png" -vcodec mpeg4 -r 1 demo.mp4
 *
 ****************************************************************************/

/* Enable POSIX extensions that are required for using
   `clock_gettime()` (used in "hpc.h"). The following `#define` must
   come at the very beginning, before including anything else. */
#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include "hpc.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int n_dims;             /* number of dimensions.                */

int n_points;           /* number of data points.               */

int n_clusters;         /* number of clusters.                  */

float *data;            /* [array of length (n_points * n_dims)]
                           `&data[i*n_dims]` points to the beginning
                           of the i-th data items, which is an array
                           of `n_dims` floating-point numbers.  */

float *centroids;       /* [array of length (n_clusters * n_dims)]
                           `&centroids[j*n_dims]` points to the
                           beginning of the j-th centroid.       */

float *new_centroids;   /* [array of length (n_clusters * n_dims)] */

float maxsqshift;       /* Maximum squared movement of a centroid in
                           one iteration. Shared: written inside a
                           `single`, read by the loop condition.  */

int *counts;            /* [array of length n_clusters] `counts[j]`
                           is the number of points in cluster j.  */

int *cluster_of;        /* [array of length n_points] `cluster_of[i]`
                           is the cluster assigned to point i.     */


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
void vzero( float *p )
{
    for (int d=0; d<n_dims; d++)
        p[d] = 0.0f;
}

/* Add vector `p2` to vector `p1`; store result in `p1`. */
void vadd( float *p1, const float *p2 )
{
    for (int d=0; d<n_dims; d++)
        p1[d] += p2[d];
}

/* Multiply each element of vector `p` of size `n_dims` by `v`. */
void vmul( float *p, float v )
{
    for (int d=0; d<n_dims; d++)
        p[d] *= v;
}

/* Copy `p2` into `p1`. */
void vcopy( float *p1, const float *p2 )
{
    for (int d=0; d<n_dims; d++)
        p1[d] = p2[d];
}

/* Compute the Euclidean squared distance of `p1` and `p2`. */
float sqdist( float *p1, float *p2 )
{
    float result = 0.0;
    for (int d=0; d<n_dims; d++) {
        result += (p1[d] - p2[d])*(p1[d] - p2[d]);
    }
    return result;
}

/******************************************************************************
 **
 ** K-Means algorithm begins here.
 **
 ******************************************************************************/

/* Linear index of row `i`, column `d` in a matrix with `n_dims`
   columns. `data[IDX(i, d)]` == `data[i*n_dims + d]`. */
int IDX(int i, int d)
{
    return i*n_dims + d;
}

/* Return a random integer in a..b. Not thread-safe (`rand()`). */
int randab(int a, int b)
{
    return a + rand() % (b-a+1);
}

/* Centroids are initialized by randomly selecting `n_clusters` data
   points (Knuth's selection algorithm, Bentley "Programming Pearls").

   DO NOT PARALLELIZE THIS FUNCTION: `rand()` is not thread-safe. */
void init_centroids( void )
{
    int select = n_clusters;
    int remaining = n_points;
    for (int i=0; (i < n_points) && (select > 0); i++) {
        if ((rand() % remaining) < select) {
            select--;
            vcopy( &centroids[IDX(select,0)], &data[IDX(i,0)] );
        }
        remaining--;
    }
}

/***
*
 assegna ogni punto al centroide più vicino.
tutti i thread entrano nella funzione e si dividono automaticamente i punti-
****/
void classify( void )
{
//azzera l'array condiviso counts. i trhead si dividono le iterazioni, la barriera implicita fa in modo che lo zeroing sia completato prima di passare al secondo ciclo
#pragma omp for
    for (int j=0; j<n_clusters; j++)
        counts[j] = 0;

//ogni thread elabora un sottoinsieme di punti. reduction crea una copia privata di counts per ciascun thread, le combina poi alla fine del ciclo
#pragma omp for reduction(+ : counts[ : n_clusters])
    for (int i=0; i<n_points; i++) {

        //cerca il centroide più vicino
        int nearest = 0;
        float mindist = sqdist( &data[IDX(i,0)], &centroids[IDX(0,0)] );
        for (int j=1; j<n_clusters; j++) {
            const float dist = sqdist( &data[IDX(i,0)], &centroids[IDX(j,0)] );
            if ( dist < mindist ) {
                mindist = dist;
                nearest = j;
            }
        }

        /* `cluster_of[i]` is written by exactly one thread (the one
           that owns iteration i): no race. `counts` is handled by the
           reduction clause. */
        cluster_of[i] = nearest;
        counts[nearest]++;
    }
    //barriera implicita, counts arrivati a questo punto contiene i valori globali e tutti i thread li vedono
}

//aggiornamento dei centroidi
void update_centroids( void )
{
    //azzeramento del buffer
#pragma omp for
    for (int j=0; j<n_clusters; j++)
        vzero(&new_centroids[IDX(j,0)]);

    //somma ogni punto nel centroide del suo cluster.
    //qui ogni thread ha una copia privata dell'intera matrice
#pragma omp for reduction(+ : new_centroids[ : n_clusters * n_dims])
    for (int i=0; i<n_points; i++) {
        vadd(&new_centroids[IDX(cluster_of[i],0)], &data[IDX(i,0)]);//new centroids contiene le somme globali

    }


#pragma omp single
    {
        float m = 0.0f;
        for (int j=0; j<n_clusters; j++) {
            if (counts[j] == 0) {
                //cluster vuoto: manteniamo il vecchio centroide
                vcopy(&new_centroids[IDX(j,0)], &centroids[IDX(j,0)]);
            } else {
              //calcolo del baricentro: somma/num dei posti
                vmul(&new_centroids[IDX(j,0)], 1.0f / counts[j]);
            }
            //calcolo dello spostamento
            const float sqshift = sqdist(&centroids[IDX(j,0)],
                                         &new_centroids[IDX(j,0)]);
            if (sqshift > m)
                m = sqshift;
            vcopy(&centroids[IDX(j,0)], &new_centroids[IDX(j,0)]);
        }
        maxsqshift = m; //variabile condivisa
    }
}

/******************************************************************************
 **
 ** Input/output functions. DO NOT parallelize them.
 **
 ******************************************************************************/

/* Read the input data from `f`. Each row must contain `n_dims`
   numbers. Initializes `n_dims` and `n_points`. */
void read_input( FILE *f )
{
    const size_t BUFLEN = 1024;
    char buffer[BUFLEN];

    char *i_dont_care = fgets(buffer, BUFLEN, f);
    (void)i_dont_care; /* Avoid a compiler warning. */
    n_dims = -1;
    char *start, *end = buffer;
    do {
        start = end;
        strtof(start, &end);
        n_dims++;
    } while (end != start);

    assert(n_dims > 0); /* If this fails, the first line is empty. */

    /* Rewind the file and count how many data items are there. */
    rewind(f);
    int n_items = 0;
    float dummy;
    while (1 == fscanf(f, "%f", &dummy))
        n_items++;

    assert(n_items % n_dims == 0); /* If this fails, some line has a
                                      number of fields != n_dims. */

    n_points = n_items / n_dims;

    data = (float*)safe_malloc(n_points * n_dims * sizeof(*data));

    /* Rewind and read the actual data. */
    rewind(f);
    for (int i=0; i<n_points; i++) {
        for (int d=0; d<n_dims; d++) {
            const int nread = fscanf(f, "%f", &data[IDX(i, d)]);
            assert(nread == 1);
        }
    }
}

#ifdef MAKE_MOVIE

/* Save the intermediate coordinates of the centroids. Enabled by
   defining `MAKE_MOVIE` at compilation time. DO NOT use when timing. */
void save_centroids( int iter )
{
    char buf[1024];

    snprintf(buf, sizeof(buf), "centroids_%03u.txt", (unsigned)iter);
    FILE *f = fopen(buf, "w"); assert(f != NULL);
    if (f == NULL) {
        fprintf(stderr, "FATAL: can not open file \"%s\" for writing\n", buf);
        exit(EXIT_FAILURE);
    }
    for (int j=0; j<n_clusters; j++) {
        for (int d=0; d<n_dims; d++) {
            fprintf(f, "%f ", centroids[IDX(j, d)]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

/* Save the intermediate coordinates of the points and their clusters.
   Enabled by defining `MAKE_MOVIE`. DO NOT use when timing. */
void save_clusters( int iter )
{
    char buf[1024];

    snprintf(buf, sizeof(buf), "clusters_%03u.txt", (unsigned)iter);
    FILE *f = fopen(buf, "w");
    if (f == NULL) {
        fprintf(stderr, "FATAL: can not open file \"%s\" for writing\n", buf);
        exit(EXIT_FAILURE);
    }
    for (int i=0; i<n_points; i++) {
        for (int d=0; d<n_dims; d++) {
            fprintf(f, "%f ", data[IDX(i, d)]);
        }
        fprintf(f, "%d\n", cluster_of[i]);
    }
    fclose(f);
}

#endif

/* Print the final result: centroid coordinates (as comments) and the
   list of data points with their cluster id. */
void save_results( FILE *f )
{
    fprintf(f, "# Centroids:\n#\n");
    for (int j=0; j<n_clusters; j++) {
        fprintf(f, "# %3d :", j);
        for (int d=0; d<n_dims; d++) {
            fprintf(f, " %f", centroids[IDX(j, d)]);
        }
        fprintf(f, "\n");
    }
    fprintf(f, "#\n");
    for (int i=0; i<n_points; i++) {
        for (int d=0; d<n_dims; d++) {
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
int main( int argc, char *argv[] )
{
    FILE *inputf, *outputf;
    const int MAXITER = 100;
    const float TOL = 1e-5;

    if (argc != 4) {
        fprintf(stderr, "Usage: %s K input_file output_file\n", argv[0]);
        return EXIT_FAILURE;
    }

    srand(123); /* Deterministic initialization of the PRNG. */

    n_clusters = atoi(argv[1]);

    if ((inputf = fopen(argv[2], "r")) == NULL) {
        fprintf(stderr, "FATAL: can not open input file \"%s\"\n", argv[2]);
        return EXIT_FAILURE;
    }

    read_input(inputf);
    fclose(inputf);

    assert(n_clusters < n_points);

    if ((outputf = fopen(argv[3], "w")) == NULL) {
        fprintf(stderr, "FATAL: can not create output file \"%s\"\n", argv[3]);
        return EXIT_FAILURE;
    }

    fprintf(outputf, "# Data points: %d\n",     n_points);
    fprintf(outputf, "# Dimensions: %d\n",      n_dims);
    fprintf(outputf, "# Clusters: %d\n",        n_clusters);

    printf("\nInput file....... %s\n", argv[2]);
    printf("Output file...... %s\n", argv[3]);
    printf("Data points (N).. %d\n", n_points);
    printf("Dimensions (D)... %d\n", n_dims);
    printf("Clusters (K)..... %d\n\n", n_clusters);

    centroids     = (float*)safe_malloc(n_clusters * n_dims * sizeof(*centroids));
    new_centroids = (float*)safe_malloc(n_clusters * n_dims * sizeof(*new_centroids));
    cluster_of    = (int*)  safe_malloc(n_points   * sizeof(*cluster_of));
    counts        = (int*)  safe_malloc(n_clusters * sizeof(*counts));

    init_centroids();

    printf("Main loop starts\n\n");

    maxsqshift = 0.0f;
    int iter = 0;
    const double tstart = hpc_gettime();

    /*
     * Unica regione parallela che racchiude l'intero ciclo
     */
#pragma omp parallel default(shared)
    {
        do {
            classify();

#ifdef MAKE_MOVIE

#pragma omp single
            {
                save_centroids(iter);
                save_clusters(iter);
            }
#endif
            //ricalcolo dei centrodici
            update_centroids();

            //avanzamento del contatore di iterazioni. fatto una sola volta e con barriera in modo che tutti i thread vedano lo stesso valore di maxqshift e devono capire come continuare
#pragma omp single
            {
                printf("Iteration %3d, maxsqshift = %f\n", iter, maxsqshift);
                iter++;
            }
        } while ( (maxsqshift > TOL*TOL) && (iter <= MAXITER) );
    }//fine regione parallela

    const double elapsed = hpc_gettime() - tstart;

    printf("\nMain loop completed\n");
    printf("Elapsed time %.3f\n\n", elapsed);

    save_results(outputf); //scrive i centroidi finali

    fclose(outputf);

    free(data);
    free(centroids);
    free(new_centroids);
    free(cluster_of);
    free(counts);

    return EXIT_SUCCESS;
}