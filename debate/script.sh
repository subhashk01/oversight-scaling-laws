
#!/bin/bash

for i in {0..19}; do
    for debater in {0..13}; do
        python debate_closed_general.py --debater $debater --data_idx $i --dataset_name truthfulqa
    done
done

for i in {0..19}; do
    for debater in {0..13}; do
        python debate_closed_general.py --debater $debater --data_idx $i --dataset_name prontoqa
    done
done


for i in {0..19}; do
    for debater in {0..13}; do
        python debate_new.py --debater $debater --data_idx $i --dataset_name quality
    done
done

for i in {0..19}; do
    for debater in {0..13}; do
        python debate_new.py --debater $debater --data_idx $i --dataset_name boolq
    done
done