#!/bin/bash
#
# benchmark.sh -- misura scalabilità (speedup ed efficienza) di omp-k-means.
#
# Uso:
#     ./benchmark.sh K input_file [n_ripetizioni]
#
# Esempio:
#     ./benchmark.sh 8 big.in 5
#
# Da lanciare sulla macchina multi-core del corso (NON su una macchina a 1 core:
# lì i tempi a piu' thread misurano solo overhead, non speedup).
#
# Nota: il tempo usato e' quello stampato dal programma ("Elapsed time"), che
# misura solo il loop di calcolo, non la lettura/scrittura dei file.

set -e

K="${1:?Uso: ./benchmark.sh K input_file [n_ripetizioni]}"
INPUT="${2:?manca input_file}"
REPS="${3:-5}"          # ripetizioni per ogni configurazione (si tiene il minimo)
BIN=./omp-k-means
SRC=omp-k-means.c

# numero massimo di thread = core fisici disponibili
MAXT=$(nproc)

# ---- compilazione ----
echo "Compilo $SRC ..."
gcc -std=c99 -Wall -Wpedantic -fopenmp -O2 "$SRC" -o "$BIN"
echo "Core disponibili: $MAXT"
echo

# estrae il campo numerico da "Elapsed time X.XXX"
get_time() { grep -i "Elapsed time" | awk '{print $3}'; }

# ---- tempo base (1 thread) come minimo su REPS run ----
declare -A best
for T in $(seq 1 "$MAXT"); do
    tmin=""
    for r in $(seq 1 "$REPS"); do
        t=$(OMP_NUM_THREADS="$T" "$BIN" "$K" "$INPUT" /dev/null | get_time)
        if [ -z "$tmin" ] || awk "BEGIN{exit !($t < $tmin)}"; then tmin="$t"; fi
    done
    best[$T]="$tmin"
done

# ---- tabella risultati ----
T1="${best[1]}"
printf "%-8s %-12s %-10s %-12s\n" "thread" "tempo(s)" "speedup" "efficienza"
printf "%-8s %-12s %-10s %-12s\n" "------" "--------" "-------" "----------"
for T in $(seq 1 "$MAXT"); do
    t="${best[$T]}"
    sp=$(awk "BEGIN{printf \"%.2f\", $T1/$t}")
    ef=$(awk "BEGIN{printf \"%.1f%%\", 100*($T1/$t)/$T}")
    printf "%-8s %-12s %-10s %-12s\n" "$T" "$t" "$sp" "$ef"
done

echo
echo "speedup = T(1)/T(p)   |   efficienza = speedup/p"
echo "Per il report: grafica speedup vs numero di thread e confrontalo con la retta ideale y=x."