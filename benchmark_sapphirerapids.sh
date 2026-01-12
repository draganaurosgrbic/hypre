#!/bin/bash
#SBATCH --job-name=hypre_sapphirerapids
#SBATCH --output=hypre_sapphirerapids_%j.log
#SBATCH --constraint=sapphirerapids
#SBATCH --partition=commons
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=04:00:00

THREADS=(1 2 4 8 16 24 32 48 64 80 96)
SIZES=("100" "160" "250" "300")

lscpu

MATRIX_BASE_PATH="/scratch/dg76/thesis_work/total/hypre-2/data"

for sz in "${SIZES[@]}"; do    
    for t in "${THREADS[@]}"; do
        export OMP_NUM_THREADS=$t
        
        if [ $t -gt 56 ]; then 
            export OMP_PROC_BIND=spread
        else 
            export OMP_PROC_BIND=close
        fi

        JOB_ID=${SLURM_JOB_ID:-"pid_$$"}
        EXEC1="/scratch/dg76/thesis_work/total/hypredrive-1/build/hypredrive"
        FILE_CSR="perf_csr_${sz}_${JOB_ID}.data"
        perf record -o "$FILE_CSR" -g -e task-clock,cycles,instructions,L1-dcache-load-misses,LLC-load-misses $EXEC1 "${MATRIX_BASE_PATH}/matrix_${sz}.yml"
        python3 parse_perf2.py 1 "$FILE_CSR"
        rm "$FILE_CSR"

        EXEC2="/scratch/dg76/thesis_work/total/hypredrive-2/build/hypredrive"
        FILE_ELL8="perf_ell8_${sz}_${JOB_ID}.data"
        perf record -o "$FILE_ELL8" -g -e task-clock,cycles,instructions,L1-dcache-load-misses,LLC-load-misses $EXEC2 "${MATRIX_BASE_PATH}/matrix_${sz}.yml"
        python3 parse_perf2.py 3 "$FILE_ELL8"
        rm "$FILE_ELL8"

    done
done

