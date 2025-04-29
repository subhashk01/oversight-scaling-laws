#!/bin/bash

for overall_num_trials in {1..3}
do
    echo "Overall trial $overall_num_trials"
    for inner_num_trials in {1..2}
    do
        for i in {0..12}
        do
            for j in {0..12}
            do
                python3 ai_control.py --openmind_llm_strength $i --openmind_monitor_llm_strength $j &
            done
        done
    done
    wait
done