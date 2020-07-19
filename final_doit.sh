#!/bin/bash
#SBATCH -p short
#SBATCH --gres gpu:1
#SBATCH --array 0-2
#SBATCH --exclude=calculon,ash,ava

##SBATCH --depend afterok:159848
##SBATCH --array 0-9
##SBATCH --qos overcap

# either run this file with sbatch doit.sh or with just ./doit.sh
# sbatch notes:
# To run lots of jobs at once use a job array with sbatch. Be sure to specify
# run.py -p batch. The sbatch command specifies its own partition (SBATCH -p)
# and this makes run.py interact correctly with sbatch.
# Double comments (##) disable the options below, which only matter for sbatch.
#
##SBATCH -p short
##SBATCH --gres gpu:1
##SBATCH --array 0-9
#
#  e.g., #SBATCH --array 0-226%10
#
# be careful... this option doesn't always work with sbatch
# -x walle,jarvis
# -w walle
# --qos overcap
EXTRA="-x calculon,ash,hal,ripl-s1,ava"

# use scontrol hold jobid to prevent a job from being scheduled without canceling
# use scontrol release jobid to allow that job to be scheduled
# use scontrol requeue jobid to make a job pending again (not sure if it kills a running job; different than suspend)
# use `scontrol update ArrayTaskThrottle=<new-max-limit> JobId=<jobid>` to change the max job limit for a running array

# NOTE: All these experiments are after the abot question length bug.
# For stage 1 this corresponds to an extra .0 on the end of exp15 models.
# For other models this uses codes exp19 (stage 2.a) and exp20 (stage 2.b)

## Abot
#explist=(
#exp2.0.0.0
#)

## Stage 1
#explist=(
#exp15.0.1.0.0
#)

## Stage 2.A
#explist=(
#exp19.0.0.0.1
#exp19.1.0.0.1
#exp19.2.0.0.1
#exp19.3.0.0.1
#exp19.1.0.1.1
#exp19.2.0.1.1
#exp19.3.0.1.1
#exp19.1.0.2.1
#exp19.2.0.2.1
#exp19.3.0.2.1
#)

## Stage 2.B
#explist=(
#exp20.0.2.0
#exp20.1.2.0
#exp20.2.2.0
#exp20.3.2.0
#exp20.1.2.1
#exp20.2.2.1
#exp20.3.2.1
#exp20.1.2.2
#exp20.2.2.2
#exp20.3.2.2
#)

## Direct to Stage 2.B
#explist=(
#exp19.0.3.0.1
#exp19.1.3.0.1
#exp19.2.3.0.1
#exp19.3.3.0.1
#exp19.1.3.1.1
#exp19.2.3.1.1
#exp19.3.3.1.1
#exp19.1.3.2.1
#exp19.2.3.2.1
#exp19.3.3.2.1
#)

## Parallel Speaker
#explist=(
#exp19.0.2.0.1
#exp19.1.2.0.1
#exp19.2.2.0.1
#exp19.3.2.0.1
#exp19.1.2.1.1
#exp19.2.2.1.1
#exp19.3.2.1.1
#exp19.1.2.2.1
#exp19.2.2.2.1
#exp19.3.2.2.1
#)

## Fine Decoder
#explist=(
#exp19.0.1.0.1
#exp19.1.1.0.1
#exp19.2.1.0.1
#exp19.3.1.0.1
#exp19.1.1.1.1
#exp19.2.1.1.1
#exp19.3.1.1.1
#exp19.1.1.2.1
#exp19.2.1.2.1
#exp19.3.1.2.1
#)

## Only Stage 2.A
#explist=(
#exp19.0.0.0.5
#exp19.1.0.0.5
#exp19.2.0.0.5
#exp19.3.0.0.5
#exp19.1.0.1.5
#exp19.2.0.1.5
#exp19.3.0.1.5
#exp19.1.0.2.5
#exp19.2.0.2.5
#exp19.3.0.2.5
#)

## Cont Stage 1
#explist=(
#exp15.1.0.0.0
#)

## Cont Stage 2.A
#explist=(
#exp19.0.0.0.3
#exp19.1.0.0.3
#exp19.2.0.0.3
#exp19.3.0.0.3
#exp19.1.0.1.3
#exp19.2.0.1.3
#exp19.3.0.1.3
#exp19.1.0.2.3
#exp19.2.0.2.3
#exp19.3.0.2.3
#)

## Cont Stage 2.B
#explist=(
#exp20.0.4.0
#exp20.1.4.0
#exp20.2.4.0
#exp20.3.4.0
#exp20.1.4.1
#exp20.2.4.1
#exp20.3.4.1
#exp20.1.4.2
#exp20.2.4.2
#exp20.3.4.2
#)

## Cont Direct to Stage 2.B
#explist=(
#exp19.0.3.0.3
#exp19.1.3.0.3
#exp19.2.3.0.3
#exp19.3.3.0.3
#exp19.1.3.1.3
#exp19.2.3.1.3
#exp19.3.3.1.3
#exp19.1.3.2.3
#exp19.2.3.2.3
#exp19.3.3.2.3
#)

## Cont Parallel Speaker
#explist=(
#exp19.0.2.0.3
#exp19.1.2.0.3
#exp19.2.2.0.3
#exp19.3.2.0.3
#exp19.1.2.1.3
#exp19.2.2.1.3
#exp19.3.2.1.3
#exp19.1.2.2.3
#exp19.2.2.2.3
#exp19.3.2.2.3
#)

## Cont Fine Decoder
#explist=(
#exp19.0.1.0.3
#exp19.1.1.0.3
#exp19.2.1.0.3
#exp19.3.1.0.3
#exp19.1.1.1.3
#exp19.2.1.1.3
#exp19.3.1.1.3
#exp19.1.1.2.3
#exp19.2.1.2.3
#exp19.3.1.2.3
#)

# NOTE: exp15 models have an extra .0 because they were trained after the bug
# fix with out a new exp code anyway

## Non-Var Cont Stage 1
#explist=(
#exp15.2.0.0.0
#)

## Non-Var Cont Stage 2.A
#explist=(
#exp19.0.0.0.4
#exp19.1.0.0.4
#exp19.2.0.0.4
#exp19.3.0.0.4
#exp19.1.0.1.4
#exp19.2.0.1.4
#exp19.3.0.1.4
#exp19.1.0.2.4
#exp19.2.0.2.4
#exp19.3.0.2.4
#)

## Non-Var Cont Stage 2.B
#explist=(
#exp20.0.5.0
#exp20.1.5.0
#exp20.2.5.0
#exp20.3.5.0
#exp20.1.5.1
#exp20.2.5.1
#exp20.3.5.1
#exp20.1.5.2
#exp20.2.5.2
#exp20.3.5.2
#)

## Non-Var Cont Direct to Stage 2.B
#explist=(
#exp19.0.3.0.4
#exp19.1.3.0.4
#exp19.2.3.0.4
#exp19.3.3.0.4
#exp19.1.3.1.4
#exp19.2.3.1.4
#exp19.3.3.1.4
#exp19.1.3.2.4
#exp19.2.3.2.4
#exp19.3.3.2.4
#)

## Non-Var Cont Parallel Speaker
#explist=(
#exp19.0.2.0.4
#exp19.1.2.0.4
#exp19.2.2.0.4
#exp19.3.2.0.4
#exp19.1.2.1.4
#exp19.2.2.1.4
#exp19.3.2.1.4
#exp19.1.2.2.4
#exp19.2.2.2.4
#exp19.3.2.2.4
#)

## Non-Var Cont Fine Decoder
#explist=(
#exp19.0.1.0.4
#exp19.1.1.0.4
#exp19.2.1.0.4
#exp19.3.1.0.4
#exp19.1.1.1.4
#exp19.2.1.1.4
#exp19.3.1.1.4
#exp19.1.1.2.4
#exp19.2.1.2.4
#exp19.3.1.2.4
#)

## All Stage 1 models
#explist=(
#exp15.0.1.0.0
#exp15.1.0.0.0
#exp15.2.0.0.0
#)

## All Non-Stage 1 models
#explist=(
#exp19.0.0.0.1
#exp19.1.0.0.1
#exp19.2.0.0.1
#exp19.3.0.0.1
#exp19.1.0.1.1
#exp19.2.0.1.1
#exp19.3.0.1.1
#exp19.1.0.2.1
#exp19.2.0.2.1
#exp19.3.0.2.1
#exp20.0.2.0
#exp20.1.2.0
#exp20.2.2.0
#exp20.3.2.0
#exp20.1.2.1
#exp20.2.2.1
#exp20.3.2.1
#exp20.1.2.2
#exp20.2.2.2
#exp20.3.2.2
#exp19.0.3.0.1
#exp19.1.3.0.1
#exp19.2.3.0.1
#exp19.3.3.0.1
#exp19.1.3.1.1
#exp19.2.3.1.1
#exp19.3.3.1.1
#exp19.1.3.2.1
#exp19.2.3.2.1
#exp19.3.3.2.1
#exp19.0.2.0.1
#exp19.1.2.0.1
#exp19.2.2.0.1
#exp19.3.2.0.1
#exp19.1.2.1.1
#exp19.2.2.1.1
#exp19.3.2.1.1
#exp19.1.2.2.1
#exp19.2.2.2.1
#exp19.3.2.2.1
#exp19.0.1.0.1
#exp19.1.1.0.1
#exp19.2.1.0.1
#exp19.3.1.0.1
#exp19.1.1.1.1
#exp19.2.1.1.1
#exp19.3.1.1.1
#exp19.1.1.2.1
#exp19.2.1.2.1
#exp19.3.1.2.1
#exp19.0.0.0.5
#exp19.1.0.0.5
#exp19.2.0.0.5
#exp19.3.0.0.5
#exp19.1.0.1.5
#exp19.2.0.1.5
#exp19.3.0.1.5
#exp19.1.0.2.5
#exp19.2.0.2.5
#exp19.3.0.2.5
#exp19.0.0.0.3
#exp19.1.0.0.3
#exp19.2.0.0.3
#exp19.3.0.0.3
#exp19.1.0.1.3
#exp19.2.0.1.3
#exp19.3.0.1.3
#exp19.1.0.2.3
#exp19.2.0.2.3
#exp19.3.0.2.3
#exp20.0.4.0
#exp20.1.4.0
#exp20.2.4.0
#exp20.3.4.0
#exp20.1.4.1
#exp20.2.4.1
#exp20.3.4.1
#exp20.1.4.2
#exp20.2.4.2
#exp20.3.4.2
#exp19.0.3.0.3
#exp19.1.3.0.3
#exp19.2.3.0.3
#exp19.3.3.0.3
#exp19.1.3.1.3
#exp19.2.3.1.3
#exp19.3.3.1.3
#exp19.1.3.2.3
#exp19.2.3.2.3
#exp19.3.3.2.3
#exp19.0.2.0.3
#exp19.1.2.0.3
#exp19.2.2.0.3
#exp19.3.2.0.3
#exp19.1.2.1.3
#exp19.2.2.1.3
#exp19.3.2.1.3
#exp19.1.2.2.3
#exp19.2.2.2.3
#exp19.3.2.2.3
#exp19.0.1.0.3
#exp19.1.1.0.3
#exp19.2.1.0.3
#exp19.3.1.0.3
#exp19.1.1.1.3
#exp19.2.1.1.3
#exp19.3.1.1.3
#exp19.1.1.2.3
#exp19.2.1.2.3
#exp19.3.1.2.3
#exp19.0.0.0.4
#exp19.1.0.0.4
#exp19.2.0.0.4
#exp19.3.0.0.4
#exp19.1.0.1.4
#exp19.2.0.1.4
#exp19.3.0.1.4
#exp19.1.0.2.4
#exp19.2.0.2.4
#exp19.3.0.2.4
#exp20.0.5.0
#exp20.1.5.0
#exp20.2.5.0
#exp20.3.5.0
#exp20.1.5.1
#exp20.2.5.1
#exp20.3.5.1
#exp20.1.5.2
#exp20.2.5.2
#exp20.3.5.2
#exp19.0.3.0.4
#exp19.1.3.0.4
#exp19.2.3.0.4
#exp19.3.3.0.4
#exp19.1.3.1.4
#exp19.2.3.1.4
#exp19.3.3.1.4
#exp19.1.3.2.4
#exp19.2.3.2.4
#exp19.3.3.2.4
#exp19.0.2.0.4
#exp19.1.2.0.4
#exp19.2.2.0.4
#exp19.3.2.0.4
#exp19.1.2.1.4
#exp19.2.2.1.4
#exp19.3.2.1.4
#exp19.1.2.2.4
#exp19.2.2.2.4
#exp19.3.2.2.4
#exp19.0.1.0.4
#exp19.1.1.0.4
#exp19.2.1.0.4
#exp19.3.1.0.4
#exp19.1.1.1.4
#exp19.2.1.1.4
#exp19.3.1.1.4
#exp19.1.1.2.4
#exp19.2.1.2.4
#exp19.3.1.2.4
#)


## analysis and visualization for non-exp15 models
#python run.py $EXTRA -m analyze -p batch ${explist[$SLURM_ARRAY_TASK_ID]} eval4.0.0.1.3
#python run.py $EXTRA -m analyze -p batch ${explist[$SLURM_ARRAY_TASK_ID]} eval4.0.1.1.3
#python run.py $EXTRA -m analyze -p batch ${explist[$SLURM_ARRAY_TASK_ID]} eval4.0.2.1.3
#python run.py $EXTRA -m analyze -p batch ${explist[$SLURM_ARRAY_TASK_ID]} eval4.0.3.1.3
##python run.py $EXTRA -m visualize -p batch "${explist[$SLURM_ARRAY_TASK_ID]}" eval5.0.0.1.3
##python run.py $EXTRA -m visualize -p batch "${explist[$SLURM_ARRAY_TASK_ID]}" eval5.0.1.1.3
##python run.py $EXTRA -m visualize -p batch "${explist[$SLURM_ARRAY_TASK_ID]}" eval5.0.2.1.3
##python run.py $EXTRA -m visualize -p batch "${explist[$SLURM_ARRAY_TASK_ID]}" eval5.0.3.1.3
#exit

## analysis and visualization for exp15 models
#python run.py $EXTRA -m analyze -p batch ${explist[$SLURM_ARRAY_TASK_ID]} eval4.1.0.1.3
#python run.py $EXTRA -m analyze -p batch ${explist[$SLURM_ARRAY_TASK_ID]} eval4.1.1.1.3
#python run.py $EXTRA -m analyze -p batch ${explist[$SLURM_ARRAY_TASK_ID]} eval4.1.2.1.3
#python run.py $EXTRA -m analyze -p batch ${explist[$SLURM_ARRAY_TASK_ID]} eval4.1.3.1.3
#python run.py $EXTRA -m analyze -p batch ${explist[$SLURM_ARRAY_TASK_ID]} eval4.2.0.1.3
#python run.py $EXTRA -m analyze -p batch ${explist[$SLURM_ARRAY_TASK_ID]} eval4.2.1.1.3
#python run.py $EXTRA -m analyze -p batch ${explist[$SLURM_ARRAY_TASK_ID]} eval4.2.2.1.3
#python run.py $EXTRA -m analyze -p batch ${explist[$SLURM_ARRAY_TASK_ID]} eval4.2.3.1.3
#python run.py $EXTRA -m analyze -p batch ${explist[$SLURM_ARRAY_TASK_ID]} eval4.3.0.1.3
#python run.py $EXTRA -m analyze -p batch ${explist[$SLURM_ARRAY_TASK_ID]} eval4.3.1.1.3
#python run.py $EXTRA -m analyze -p batch ${explist[$SLURM_ARRAY_TASK_ID]} eval4.3.2.1.3
#python run.py $EXTRA -m analyze -p batch ${explist[$SLURM_ARRAY_TASK_ID]} eval4.3.3.1.3
#python run.py $EXTRA -m analyze -p batch ${explist[$SLURM_ARRAY_TASK_ID]} eval4.4.0.1.3
#python run.py $EXTRA -m analyze -p batch ${explist[$SLURM_ARRAY_TASK_ID]} eval4.4.1.1.3
#python run.py $EXTRA -m analyze -p batch ${explist[$SLURM_ARRAY_TASK_ID]} eval4.4.2.1.3
#python run.py $EXTRA -m analyze -p batch ${explist[$SLURM_ARRAY_TASK_ID]} eval4.4.3.1.3
##python run.py $EXTRA -m visualize -p batch "${explist[$SLURM_ARRAY_TASK_ID]}" eval5.1.0.1.3
##python run.py $EXTRA -m visualize -p batch "${explist[$SLURM_ARRAY_TASK_ID]}" eval5.1.1.1.3
##python run.py $EXTRA -m visualize -p batch "${explist[$SLURM_ARRAY_TASK_ID]}" eval5.1.2.1.3
##python run.py $EXTRA -m visualize -p batch "${explist[$SLURM_ARRAY_TASK_ID]}" eval5.1.3.1.3
##python run.py $EXTRA -m visualize -p batch "${explist[$SLURM_ARRAY_TASK_ID]}" eval5.2.0.1.3
##python run.py $EXTRA -m visualize -p batch "${explist[$SLURM_ARRAY_TASK_ID]}" eval5.2.1.1.3
##python run.py $EXTRA -m visualize -p batch "${explist[$SLURM_ARRAY_TASK_ID]}" eval5.2.2.1.3
##python run.py $EXTRA -m visualize -p batch "${explist[$SLURM_ARRAY_TASK_ID]}" eval5.2.3.1.3
##python run.py $EXTRA -m visualize -p batch "${explist[$SLURM_ARRAY_TASK_ID]}" eval5.3.0.1.3
##python run.py $EXTRA -m visualize -p batch "${explist[$SLURM_ARRAY_TASK_ID]}" eval5.3.1.1.3
##python run.py $EXTRA -m visualize -p batch "${explist[$SLURM_ARRAY_TASK_ID]}" eval5.3.2.1.3
##python run.py $EXTRA -m visualize -p batch "${explist[$SLURM_ARRAY_TASK_ID]}" eval5.3.3.1.3
##python run.py $EXTRA -m visualize -p batch "${explist[$SLURM_ARRAY_TASK_ID]}" eval5.4.0.1.3
##python run.py $EXTRA -m visualize -p batch "${explist[$SLURM_ARRAY_TASK_ID]}" eval5.4.1.1.3
##python run.py $EXTRA -m visualize -p batch "${explist[$SLURM_ARRAY_TASK_ID]}" eval5.4.2.1.3
##python run.py $EXTRA -m visualize -p batch "${explist[$SLURM_ARRAY_TASK_ID]}" eval5.4.3.1.3
#exit

## training
#python run.py $EXTRA -m train -p batch "${explist[$SLURM_ARRAY_TASK_ID]}"
#exit

## check
#python run.py $EXTRA -m check -p batch "${explist[$SLURM_ARRAY_TASK_ID]}"
#exit
