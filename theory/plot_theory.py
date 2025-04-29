# %%
import pickle
import matplotlib.pyplot as plt
import numpy as np  

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 5,
    'axes.labelsize': 7.25,
    'axes.titlesize': 7.25,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 5.5,
})

# %%
def plot_theory(monitor_slope, ai_slope, ax1=None, ax2=None, ax3=None, save_fig=True):
    res = pickle.load(open(f"data/params_to_win_prob_and_num_steps_monitor_slope_{monitor_slope}_ai_slope_{ai_slope}.pkl", "rb"))
    monitor_slope = res["monitor_slope"]
    ai_slope = res["ai_slope"]
    ai_general_intelligence_elos = res["ai_general_intelligence_elos"]
    ai_game_elos = res["ai_game_elos"]
    win_prob_matrix = res["win_prob_matrix"]
    num_steps_matrix = res["num_steps_matrix"]

    # Only a function of game ELO
    naive_win_prob_matrix = 1 / (1 + 10**((ai_game_elos) / 400))
    
    # Create figure with subplots if not provided
    if ax1 is None or ax2 is None or ax3 is None:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(5.5, 1.5), sharey=True)
        create_new_figure = True
    else:
        create_new_figure = False
        # Turn off ticks if not creating a new figure
        for ax in [ax1, ax2, ax3]:
            ax.set_xticks([])
            ax.set_yticks([])

    # Plot 1: Log odds of optimal win probability
    odds_matrix = win_prob_matrix / (1 - win_prob_matrix + 1e-10)
    log_odds_matrix = np.log(odds_matrix)
    abs_max = np.max(np.abs(log_odds_matrix))
    im1 = ax1.pcolormesh(ai_general_intelligence_elos, 
                         ai_game_elos, 
                         log_odds_matrix.T, 
                         shading='auto', 
                         cmap='RdBu', 
                         norm=plt.Normalize(vmin=-abs_max, vmax=abs_max))
    if create_new_figure:
        fig.colorbar(im1, ax=ax1)
        ax1.set_xlabel('$\\Delta$ General ELO')
        ax1.set_ylabel('$\\Delta$ Domain ELO')
        ax1.set_title('Log Odds of\nOptimal Guard Win')

    # Plot 2: Optimal number of steps
    max_steps = np.max(num_steps_matrix)
    im2 = ax3.pcolormesh(
        ai_general_intelligence_elos, 
        ai_game_elos, 
        num_steps_matrix.T, 
        shading='auto',
        cmap="Greens",
        vmin=1,
        vmax=max_steps
    )
    if create_new_figure:
        cbar = fig.colorbar(im2, ax=ax3)
        max_num_ticks = 15
        step = max(1, (max_steps + max_num_ticks) // max_num_ticks)
        ticks = np.arange(1, max_steps + 1, step)
        cbar.set_ticks(ticks)
        ax3.set_xlabel('$\\Delta$ General ELO')
        ax3.set_title('Optimal # of steps $n^*$')

    # # Plot 3: Log odds of naive win probability
    naive_win_prob_2d = np.tile(naive_win_prob_matrix, (len(ai_general_intelligence_elos), 1)).T
    naive_odds_matrix = naive_win_prob_2d / (1 - naive_win_prob_2d + 1e-10)
    naive_log_odds_matrix = np.log(naive_odds_matrix)

    # Plot 4: Difference between optimal and naive log odds
    log_odds_diff = log_odds_matrix - naive_log_odds_matrix.T
    abs_max_diff = np.max(np.abs(log_odds_diff))
    im4 = ax2.pcolormesh(
        ai_general_intelligence_elos,
        ai_game_elos,
        log_odds_diff.T,
        shading='auto',
        cmap='RdBu',
        norm=plt.Normalize(vmin=-abs_max_diff, vmax=abs_max_diff)
    )
    if create_new_figure:
        fig.colorbar(im4, ax=ax2)
        ax2.set_xlabel('$\\Delta$ General ELO')
        ax2.set_title('Optimal and One Step \n Log Odds Difference')

    # Add reference lines to all plots
    for ax in [ax1, ax2, ax3]:
        # Line with slope 1
        slope = 1
        x = ai_general_intelligence_elos
        y = x * slope
        # Filter out values where y is greater than max(ai_game_elos)
        x_filtered = x[y < max(ai_game_elos)]
        y_filtered = y[y < max(ai_game_elos)]
        # ax.plot(x_filtered, y_filtered, ':', color='black', alpha=1)
        
        # Line with asymptote slope
        asymptote_slope = (monitor_slope + ai_slope) / 2
        y = x * asymptote_slope
        x_filtered = x[y < max(ai_game_elos)]
        y_filtered = y[y < max(ai_game_elos)]
        ax.plot(x_filtered, y_filtered, ':', color='black', alpha=1)

    if create_new_figure and save_fig:
        plt.subplots_adjust(wspace=0.2)
        plt.savefig(f"plots/optimal_strategy_monitor_slope_{monitor_slope}_ai_slope_{ai_slope}.png", dpi=500, bbox_inches='tight')
        plt.show()

    return ax1, ax2, ax3

# %%


def plot_all_combinations(min_slope=1, max_slope=5):
    num_slopes = max_slope - min_slope + 1
    fig = plt.figure(figsize=(5.5, 4))
    
    # Calculate total columns with spacing
    total_cols = num_slopes * 3 + (num_slopes - 1)  # 3 plots per combination + 1 space after each set except last
    
    for monitor_slope in range(min_slope, max_slope + 1):
        for ai_slope in range(min_slope, max_slope + 1):
            row_idx = monitor_slope - min_slope
            col_idx = ai_slope - min_slope
            
            # Calculate position with spacing after every set of 3 plots
            # Each combination takes 3 columns, and we add spacing after each combination
            base_col = col_idx * 4  # 3 plots + 1 space
            
            # Create three subplots for each combination with spacing between sets
            ax1 = fig.add_subplot(num_slopes, total_cols, row_idx * total_cols + base_col + 1)
            ax2 = fig.add_subplot(num_slopes, total_cols, row_idx * total_cols + base_col + 2)
            ax3 = fig.add_subplot(num_slopes, total_cols, row_idx * total_cols + base_col + 3)
            
            # Plot with the specific axes and don't save individual figures
            plot_theory(monitor_slope, ai_slope, ax1, ax2, ax3, save_fig=False)
            
            # Add titles for each row and column
            if col_idx == 0:
                ax1.set_ylabel(f"Guard Slope = {monitor_slope}")
            if row_idx == 0:
                ax2.set_title(f"Houdini Slope = {ai_slope}")
            
            # Add visual separation between sets
            for ax in [ax1, ax2, ax3]:
                ax.spines['top'].set_visible(True)
                ax.spines['right'].set_visible(True)
                ax.spines['bottom'].set_visible(True)
                ax.spines['left'].set_visible(True)
                ax.spines['top'].set_linewidth(1.5)
                ax.spines['right'].set_linewidth(1.5)
                ax.spines['bottom'].set_linewidth(1.5)
                ax.spines['left'].set_linewidth(1.5)
    
    plt.subplots_adjust(wspace=0.4, hspace=0.3)
    plt.savefig(f"plots/all_combinations_min_{min_slope}_max_{max_slope}.png", dpi=500, bbox_inches='tight')
    plt.show()

# Uncomment to generate the giant plot
plot_all_combinations(min_slope=1, max_slope=3)

# %%


# Plot individual figures
for monitor_slope in range(1, 6):
    for ai_slope in range(1, 6):
        plot_theory(monitor_slope, ai_slope)

# %%

def plot_pareto_frontier(monitor_slope, ai_slope):
    res = pickle.load(open(f"data/params_to_win_prob_and_num_steps_monitor_slope_{monitor_slope}_ai_slope_{ai_slope}.pkl", "rb"))
    ai_general_intelligence_elos = res["ai_general_intelligence_elos"]
    ai_game_elos = res["ai_game_elos"]

    # Find points where num_steps is 0
    num_steps_matrix = res["num_steps_matrix"]
    pareto_game_elos = []
    pareto_iq_elos = []

    # For each IQ ELO value
    for i, iq_elo in enumerate(ai_general_intelligence_elos):
        # Find all game ELOs where num_steps is 1 for this IQ ELO
        zero_step_indices = np.where(num_steps_matrix[i,:] == 1)[0]
        
        if len(zero_step_indices) > 0:
            # Get the minimum game ELO where num_steps is 1
            min_game_elo_idx = np.min(zero_step_indices)
            pareto_game_elos.append(ai_game_elos[min_game_elo_idx])
            pareto_iq_elos.append(iq_elo)

    # Create a scatter plot of the Pareto frontier
    plt.figure(figsize=(8, 6))
    plt.scatter(pareto_iq_elos, pareto_game_elos, label='Pareto Frontier', s=1)
    plt.xlabel('$\\Delta$ General ELO')

    # Find minimum slope k that lies above all points
    slopes = np.array([game_elo/iq_elo for game_elo, iq_elo in zip(pareto_game_elos, pareto_iq_elos)])
    min_slope_above = np.max(slopes)

    # Plot the line y = kx
    x = np.array(ai_general_intelligence_elos)
    y = min_slope_above * x
    plt.plot(x, y, ':', color='red', alpha=0.8, label=f'y = {min_slope_above:.2f}x')
    plt.ylabel('$\\Delta$ Domain ELO')
    plt.legend()
    plt.grid(True)
    plt.show()

    return min_slope_above


monitor_and_ai_slope_to_min_slope_above = {}
for monitor_slope in range(1, 6):
    for ai_slope in range(1, 6):
        monitor_and_ai_slope_to_min_slope_above[(monitor_slope, ai_slope)] = plot_pareto_frontier(monitor_slope, ai_slope)

# %%

# Create a matrix from the dictionary
min_slopes = np.zeros((5, 5))
for (monitor_slope, ai_slope), slope in monitor_and_ai_slope_to_min_slope_above.items():
    min_slopes[monitor_slope-1, ai_slope-1] = slope

plt.figure(figsize=(8, 6))
plt.imshow(min_slopes, origin='lower')
plt.colorbar(label='Minimum slope above Pareto frontier')
plt.xlabel('AI slope')
plt.ylabel('Monitor slope') 
plt.title('Minimum slope above Pareto frontier\nfor different AI and Monitor slopes')

# Add value annotations on each cell
for i in range(5):
    for j in range(5):
        plt.text(j, i, f'{min_slopes[i,j]:.2f}', 
                ha='center', va='center')

# Set ticks to show actual slope values (1-5) instead of array indices (0-4)
plt.xticks(range(5), range(1,6))
plt.yticks(range(5), range(1,6))

plt.show()


# Functional form: min_slope_above = |monitor_slope + ai_slope| / 2
# %%
