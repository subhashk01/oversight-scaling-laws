for monitor_slope in 1 2 3 4 5; do
    for ai_slope in 1 2 3 4 5; do
        python3 experiment_with_optimal_strat.py --monitor_slope $monitor_slope --ai_slope $ai_slope &
    done
done

wait

