#!/bin/bash


set -e

K="${1:?Uso: ./benchmark-mpi.sh K input_file [n_ripetizioni] [max_processi]}"
INPUT="${2:?manca input_file}"
REPS="${3:-5}"          # ripetizioni per ogni configurazione
MAXP="${4:-12}"         # numero massimo di processi (12 = core del server)
BIN=./mpi-k-means
SRC=mpi-k-means.c
OUT=/dev/null           # l'output del clustering non ci serve qui

# ---- compilazione ----
echo "Compilo $SRC ..."
mpicc -std=c99 -Wall -Wpedantic -O2 "$SRC" -o "$BIN"
echo "Processi testati: da 1 a $MAXP,  ripetizioni: $REPS,  K=$K,  input=$INPUT"
echo

# estrae il numero da "Execution time X.XXX"
get_time() { grep -i "Execution time" | awk '{print $3}'; }

# ---- misura: per ogni P, tieni il minimo su REPS run ----
declare -A best
for P in $(seq 1 "$MAXP"); do
    tmin=""
    for r in $(seq 1 "$REPS"); do
        t=$(mpirun -n "$P" "$BIN" "$K" "$INPUT" "$OUT" | get_time)
        if [ -z "$tmin" ] || awk "BEGIN{exit !($t < $tmin)}"; then tmin="$t"; fi
    done
    best[$P]="$tmin"
    printf "  n=%-2s  t=%s s\n" "$P" "$tmin"
done

echo
T1="${best[1]}"
printf "%-8s %-12s %-16s %-12s\n" "proc" "tempo(s)" "speedup_rel" "efficienza"
printf "%-8s %-12s %-16s %-12s\n" "----" "--------" "-----------" "----------"
for P in $(seq 1 "$MAXP"); do
    t="${best[$P]}"
    sp=$(awk "BEGIN{printf \"%.3f\", $T1/$t}")
    ef=$(awk "BEGIN{printf \"%.1f%%\", 100*($T1/$t)/$P}")
    printf "%-8s %-12s %-16s %-12s\n" "$P" "$t" "$sp" "$ef"
done

echo
echo "speedup relativo = T(1)/T(p)   |   efficienza = speedup/p"
echo "Per la relazione: metti questa tabella e grafica speedup vs n. processi,"
echo "confrontandolo con la retta ideale y = x."