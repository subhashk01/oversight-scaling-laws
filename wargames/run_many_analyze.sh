#!/bin/bash

for i in {1..10}
do
    python3 analyze_new_wargame.py &
done

wait
