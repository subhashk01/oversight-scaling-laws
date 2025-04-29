#!/bin/bash

for i in {1..5}
do
    python analyze_strategies.py &
done

wait