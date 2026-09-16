## Autore: Rigato Valentina 1027266

## This Makefile can be used to compile the parallel versions of the
## K-Means program. The following targets are available:
##
## clean         remove temporary files
## init          crea struttura cartelle necessaria
##
## compile.*     compila il codice
## run.*         esegue il codice
##
## *.inputgen    genera file input
## *.omp         versione omp dell'algoritmo k-mean
## *.mpi         versione mpi dell'algoritmo k-mean
##
## Guida:
## 1) make init per generare le cartelle
## 2) make compile.inputgen
## 3) make run.inputgen
## 4) make compile.omp
## 5) make run.omp OMP_THREADS=12
## 6) make compile.mpi
## 7) make run.mpi NPROC=12

CFLAGS += -std=c99 -Wall -Wpedantic
MPICC ?= mpicc
EXES := omp-k-means mpi-k-means inputgen

.PHONY: demo init clean run omprun inputgen mpirun compile

BIN_DIR = bin

OMP_THREADS ?= 12 # valore di default se non specificato
NPROC ?= 4 # valore di default se non specificato

OUTPUT_DIR = output
OUTPUT_NAME = output.out
OUTPUT_FILE = $(OUTPUT_DIR)/$(OUTPUT_NAME)

INPUT_DIR = input
# INPUT_NAME = test-N150000-D20-K5.in
INPUT_FILE = $(INPUT_DIR)/$(INPUT_NAME)

# punti
N = 150000
# dimensioni
D = 20
# cluster
K = 5

INPUT_NAME = test-N$(N)-D$(D)-K$(K).in

ALL: $(EXES)

## Genera struttura cartelle
init:
	mkdir -p bin input libs output

compile.inputgen:
	$(CC) $(CFLAGS) inputgen.c -o $(BIN_DIR)/inputgen

inputgen: inputgen-bin

## Generazione .in
run.inputgen:
	./$(BIN_DIR)/inputgen $(N) $(D) $(K) > $(INPUT_DIR)/test-N$(N)-D$(D)-K$(K).in

## OMP
compile.omp:
	gcc omp-k-means.c -Wall -Wpedantic -std=c99 -fopenmp -o $(BIN_DIR)/omp-k-means

run.omp:
	OMP_NUM_THREADS=$(OMP_THREADS) $(BIN_DIR)/omp-k-means $(K) $(INPUT_FILE) $(OUTPUT_FILE)

## MPI
compile.mpi:
	$(MPICC) mpi-k-means.c -Wall -Wpedantic -std=c99 -o $(BIN_DIR)/mpi-k-means

run.mpi:
	mpirun -n $(NPROC) $(BIN_DIR)/mpi-k-means $(K) $(INPUT_FILE) $(OUTPUT_FILE)

clean:
	\rm -f $(EXES) *.o *~ centroids_*.txt clusters_*.txt img_*.png demo.out demo.mp4
	rm -f $(OUTPUT_DIR)/*.out $(OUTPUT_DIR)/*.txt
	rm -f $(INPUT_DIR)/*.out $(INPUT_DIR)/*.txt