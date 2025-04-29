# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from decimal import Decimal, getcontext
import argparse

# %%

parser = argparse.ArgumentParser()
parser.add_argument("--monitor_slope", type=float, default=1)
parser.add_argument("--ai_slope", type=float, default=1)
args = parser.parse_args()

monitor_slope = args.monitor_slope
ai_slope = args.ai_slope

# %%

# Set precision for Decimal calculations
getcontext().prec = 80

# Normalize to elo slope of monitor is 1 and starts at 0

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 5,
    'axes.labelsize': 7.25,
    'axes.titlesize': 7.25,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 5.5,
})


def plot(monitor_slope, ai_slope, ai_intercept, ai_g_elo):
    xs = np.linspace(0, ai_g_elo, 100)
    ai_ys = ai_slope * xs + ai_intercept
    monitor_ys = monitor_slope * xs
    plt.plot(xs, ai_ys, color='#ff6666', label='AI', linewidth=2)
    plt.plot(xs, monitor_ys, color='#3366ff', label='Monitor', linewidth=2)
    plt.plot(0, 0, marker='o', color='#3366ff', markersize=10, label='Starting Monitor')
    plt.plot(ai_g_elo, ai_slope * ai_g_elo + ai_intercept, marker='o', color='#ff6666', markersize=10, label='Ending AI')
    plt.show()


def win_probability_player_a(elo_a, elo_b):
    elo_a = Decimal(str(elo_a))
    elo_b = Decimal(str(elo_b))
    return Decimal('1') / (Decimal('1') + Decimal('10') ** ((elo_b - elo_a) / Decimal('400')))


def win_probability(monitor_slope, ai_slope, ai_intercept, ai_g_elo, num_steps):
    g_elos = [Decimal(str(g)) for g in np.linspace(0, ai_g_elo, num_steps + 1)]
    monitor_elos = [g * Decimal(str(monitor_slope)) for g in g_elos]
    
    # Convert numpy arrays to lists of Decimals
    g_elos_dec = [Decimal(str(g)) for g in g_elos]
    monitor_elos = [g * Decimal(str(monitor_slope)) for g in g_elos_dec]
    ai_elos = [g * Decimal(str(ai_slope)) + Decimal(str(ai_intercept)) for g in g_elos_dec]
    
    monitor_elos = monitor_elos[:-1]
    ai_elos = ai_elos[1:]
    
    win_probs = [win_probability_player_a(monitor_elos[i], ai_elos[i]) for i in range(len(ai_elos))]

    for p in win_probs:
        assert p < 1

    total_win_prob = Decimal('1')
    for prob in win_probs:
        total_win_prob *= prob
    
    return total_win_prob


n = 2000
d = 1000
num_steps = 20
print((d - n) + n / num_steps)
print(win_probability(1, 1, d - n, n, num_steps))
# %%

def plot_optimal_strategy(ax, monitor_slope, ai_slope, ai_intercept, ai_g_elo, plot_y_axis=True):
    # Find optimal number of steps
    best_num_steps = 0
    best_win_prob = 0
    for num_steps in np.arange(1, 20):
        win_prob = win_probability(monitor_slope, ai_slope, ai_intercept, ai_g_elo, num_steps)
        if win_prob > best_win_prob:
            best_win_prob = win_prob
            best_num_steps = num_steps

    # Plot the curves
    xs = np.linspace(0, ai_g_elo, 100)
    ai_ys = ai_slope * xs + ai_intercept
    monitor_ys = monitor_slope * xs
    ax.plot(xs, ai_ys, color='#ff6666', label='Houdini ELO Curve', linewidth=2)
    ax.plot(xs, monitor_ys, color='#3366ff', label='Guard ELO Curve', linewidth=2)
    
    # Plot the optimal strategy points
    g_elos = np.linspace(0, ai_g_elo, best_num_steps + 1)
    monitor_elos = g_elos * monitor_slope
    ai_elos = g_elos * ai_slope + ai_intercept
    
    # Plot lines between curves
    for i in range(best_num_steps):
        if i != 0:  # Middle point
            ax.plot([g_elos[i], g_elos[i+1]], [monitor_elos[i], ai_elos[i+1]], 
                     color='#33cc33', linestyle='--', linewidth=1)
            # Add vertical connection
            ax.plot([g_elos[i], g_elos[i]], [monitor_elos[i], ai_elos[i]], 
                        color='#9933ff', linestyle=':', linewidth=1, label="Transfer Step" if i == 1 else "")
        else:
            ax.plot([g_elos[i], g_elos[i+1]], [monitor_elos[i], ai_elos[i+1]], 
                     color='#33cc33', linestyle='--', linewidth=1, label='Game Step' if i == 0 else "")
    
    # Plot points
    for i in range(len(g_elos)-1):
        ax.plot(g_elos[i], monitor_elos[i], 'o', color="#3366ff", markersize=4)
        ax.plot(g_elos[i+1], ai_elos[i+1], 'o', color="#ff6666", markersize=4)

    ax.plot(0, 0, '*', color='#3366ff', markersize=10, label='Starting Guard')
    ax.plot(ai_g_elo, ai_slope * ai_g_elo + ai_intercept, '*', 
             color='#ff6666', markersize=10, label='Target Houdini')
    
    # ax.set_title(f'Guard ELO Slope: {monitor_slope}, Houdini ELO Slope: {ai_slope}\nHoudini ELO Intercept: {ai_intercept}, Houdini ELO Starting Advantage: {ai_g_elo}\nWin Probability: {best_win_prob:.3f}')
    ax.set_title(f"$n^*$ = {best_num_steps}")
    ax.set_xlabel('General ELO')
    if plot_y_axis:
        ax.set_ylabel('Domain ELO')
    ax.grid(True, alpha=0.3)

    # Add extra space to the x and y axis
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_range = x_max - x_min
    y_range = y_max - y_min
    ax.set_xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)

# Create a figure with 4 subplots
fig, axs = plt.subplots(1, 4, figsize=(5.5, 2))
axs = axs.flatten()

# Plot different parameter combinations 
combos = [
    (1, 1, 400, 1200),
    (1, 1, -200, 500),
    (1, 2, -800, 1000),
    (1, 3, -2000, 1000)
]

for i, combo in enumerate(combos):
    plot_optimal_strategy(axs[i], *combo, plot_y_axis=i == 0)

# Create a single legend for the entire figure
handles, labels = axs[0].get_legend_handles_labels()
plt.tight_layout(pad=0)
fig.legend(handles, labels, ncol=3, bbox_to_anchor=(0.5, 1.1), loc='center')
plt.savefig(f"plots/optimal_strategy_examples.pdf", bbox_inches='tight')
plt.show()
# %%


# Change to be parameterized by:
# Initial houdini general intelligence ELO advantage
# Initial houdini game ELO advantage
# %%


def run_and_save_optimal_strategy(monitor_slope, ai_slope, max_game_elo=2000, max_iq_elo=2000, step=10):
    params_to_win_prob_and_num_steps = {}
    ai_game_elos = np.arange(-max_game_elo, max_game_elo, step)
    ai_general_intelligence_elos = np.arange(0, max_iq_elo, step)
    for ai_game_elo in tqdm(ai_game_elos):
        for ai_general_intelligence_elo in ai_general_intelligence_elos:
            best_num_steps = 0
            best_win_prob = 0
            ai_intercept = ai_game_elo - ai_general_intelligence_elo * ai_slope

            num_steps = 0
            while num_steps < 20 or best_num_steps == num_steps:
                num_steps += 1
                win_prob = win_probability(
                    monitor_slope, 
                    ai_slope, 
                    ai_intercept, 
                    ai_general_intelligence_elo, 
                    num_steps
                )
                if win_prob > best_win_prob:
                    best_win_prob = win_prob
                    best_num_steps = num_steps

            params_to_win_prob_and_num_steps[(ai_general_intelligence_elo, ai_game_elo)] = (best_win_prob, best_num_steps)


    # Create 2D arrays for win prob and num steps
    win_prob_matrix = np.zeros((len(ai_general_intelligence_elos), len(ai_game_elos)))
    num_steps_matrix = np.zeros((len(ai_general_intelligence_elos), len(ai_game_elos)))

    for i, ai_general_intelligence_elo in enumerate(ai_general_intelligence_elos):
        for j, ai_game_elo in enumerate(ai_game_elos):
            win_prob, num_steps = params_to_win_prob_and_num_steps[(ai_general_intelligence_elo, ai_game_elo)]
            win_prob_matrix[i,j] = win_prob
            num_steps_matrix[i,j] = num_steps

    res = {
        "params_to_win_prob_and_num_steps": params_to_win_prob_and_num_steps,
        "monitor_slope": monitor_slope,
        "ai_slope": ai_slope,
        "ai_general_intelligence_elos": ai_general_intelligence_elos,
        "ai_game_elos": ai_game_elos,
        "win_prob_matrix": win_prob_matrix,
        "num_steps_matrix": num_steps_matrix,
    }

    pickle.dump(res, open(f"data/params_to_win_prob_and_num_steps_monitor_slope_{monitor_slope}_ai_slope_{ai_slope}.pkl", "wb"))

run_and_save_optimal_strategy(monitor_slope, ai_slope)
# %%


