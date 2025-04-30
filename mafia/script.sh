#!/bin/bash
#SBATCH -t 23:59:00

for i in {1..20}; do
    for villager in {0..13}; do
        for mafia in {0..13}; do
            python Mafia.py --villager $villager --mafia $mafia
        done
    done
done