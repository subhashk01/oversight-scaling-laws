# %%
import numpy as np
from scipy.optimize import minimize, curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import os

# %%
'''
MODEL_LIST = [
    "openai/chatgpt-4o-latest",
    "google/gemini-2.0-flash-001",
    "google/gemini-2.0-flash-lite-001",
    "google/gemini-flash-1.5",
    "meta-llama/llama-3.3-70b-instruct",
    "anthropic/claude-3.5-haiku",
    "google/gemma-2-27b-it",
    "meta-llama/llama-3-70b-instruct",
    "google/gemma-2-9b-it",
    "qwen/qwen-2-72b-instruct",
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3-8b-instruct",
    "google/gemini-pro",
    "meta-llama/llama-3.2-3b-instruct"
]

MODEL_NAME_ONLY = [MODEL.split("/")[-1] for MODEL in MODEL_LIST]

ELO_LIST = [
    1377,
    1356,
    1310,
    1271,
    1255,
    1236,
    1220,
    1207,
    1192,
    1187,
    1176,
    1152,
    1131,
    1103,
]

xerr_neg = [6, 5, 6, 3, 3, 5, 3, 2, 3, 3, 3, 3, 5, 8]
xerr_pos = [5, 6, 6, 3, 5, 5, 3, 2, 3, 3, 3, 2, 4, 6]

elos = {}
elos['llm_name'] = MODEL_NAME_ONLY
elos['elo'] = ELO_LIST
elos['xerr_neg'] = xerr_neg
elos['xerr_pos'] = xerr_pos

pd.DataFrame(elos).to_csv('llm_elos.csv', index=False)
'''

# %%
import numpy as np
import lmfit

def _elo_expected_score(rating_i, rating_j):
    """
    Given two Elo ratings rating_i and rating_j,
    returns the expected probability that 'i' wins against 'j'.
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_j - rating_i) / 400.0))

########################################################
# 2) Negative log-likelihood for rectangular M x N data
########################################################

def _neg_log_likelihood_binomial_rect(elo_vector, win_rate_matrix, num_matrix):
    eps = 1e-15
    M = win_rate_matrix.shape[0]
    N = win_rate_matrix.shape[1]

    nll = 0.0
    for i in range(M):
        rating_i = elo_vector[i]  # row player i
        for j in range(N):
            if num_matrix[i, j] is None or num_matrix[i, j] == 0:
                # no data here
                continue
            rating_j = elo_vector[M + j]  # column player j

            frac_ij = win_rate_matrix[i, j]
            games_ij = num_matrix[i, j]

            # logistic-based expected probability that i beats j
            p_ij = _elo_expected_score(rating_i, rating_j)

            # binomial negative log-likelihood
            nll -= games_ij * (
                frac_ij * np.log(p_ij + eps) +
                (1.0 - frac_ij) * np.log(1.0 - p_ij + eps)
            )
    return nll


def compute_elo_ratings_binomial_rect(win_rate_matrix, num_matrix, initial_guess=None, method='L-BFGS-B'):
    """
    Fits Elo ratings for the *rectangular* scenario:
      - M row-players, N column-players
      - M + N total rating parameters
      - For each (i,j), row i's win fraction over column j is given.

    Returns:
      row_elos (length M), col_elos (length N), res (scipy OptimizeResult)
    """
    M = win_rate_matrix.shape[0]
    N = win_rate_matrix.shape[1]
    if initial_guess is None:
        initial_guess = np.zeros(M + N)

    res = minimize(
        fun=_neg_log_likelihood_binomial_rect,
        x0=initial_guess,
        args=(win_rate_matrix, num_matrix),
        method=method
    )

    if not res.success:
        print("Warning: Binomial-based Elo optimization did not converge successfully.")

    final_elos = res.x
    row_elos = final_elos[:M]
    col_elos = final_elos[M:]
    return row_elos, col_elos, res

########################################################
# 4) Parametric bootstrap in rectangular case
########################################################


def bootstrap_elo_confidence_intervals_rect(estimated_elos, num_matrix, initial_guess=None,
                                            n_samples=200, confidence=0.95,
                                            random_seed=0):
    """
    Parametric bootstrap for the rectangular scenario:
      - We treat each cell (i, j) as Binomial(N_{i,j}, w_{i,j}).

    For each bootstrap replicate:
      1. For each (i,j), sample X_{i,j}^* ~ Binomial(N_{i,j}, w_{i,j}),
         define w_{i,j}^* = X_{i,j}^* / N_{i,j}.
      2. Re-fit Elo for that replicate.
      3. Collect the (M + N) Elo ratings.

    At the end, return percentile-based confidence intervals for each parameter.
    """
    np.random.seed(random_seed)
    M = num_matrix.shape[0]
    N = num_matrix.shape[1]
    n_params = M + N

    all_elos = []  # store each replicate's entire parameter vector

    for _ in range(n_samples):
        # create a new fraction matrix
        M_boot = np.zeros_like(num_matrix, dtype=float)
        for i in range(M):
            for j in range(N):
                if num_matrix[i, j] is None or num_matrix[i, j] == 0:
                    M_boot[i, j] = 0.0
                    continue
                frac_ij = _elo_expected_score(estimated_elos[i], estimated_elos[M + j])
                games_ij = num_matrix[i, j]
                x_ij_star = np.random.binomial(games_ij, frac_ij)
                M_boot[i, j] = x_ij_star / float(games_ij)

        row_elos_b, col_elos_b, _ = compute_elo_ratings_binomial_rect(M_boot, num_matrix, initial_guess)
        all_elos.append(np.concatenate([row_elos_b, col_elos_b]))

    all_elos = np.array(all_elos)  # shape = (n_samples, M+N)

    # point estimate is the mean
    param_means = np.mean(all_elos, axis=0)

    # percentile-based intervals
    alpha = 1.0 - confidence
    lower_q = 100.0 * alpha / 2.0
    upper_q = 100.0 * (1.0 - alpha / 2.0)

    param_lower = np.percentile(all_elos, lower_q, axis=0)
    param_upper = np.percentile(all_elos, upper_q, axis=0)

    # separate them back out
    row_elos_mean = param_means[:M]
    col_elos_mean = param_means[M:]

    row_elos_lower = param_lower[:M]
    row_elos_upper = param_upper[:M]
    col_elos_lower = param_lower[M:]
    col_elos_upper = param_upper[M:]

    return (row_elos_mean, row_elos_lower, row_elos_upper,
            col_elos_mean, col_elos_lower, col_elos_upper)

def translation(domain_elos):
    # translates domain_elos to have same mean as general elos
    real_elo_mean = np.mean(get_llm_elos()['elo'])
    return real_elo_mean - np.mean(domain_elos)


def get_elo_bounds(win_rate_matrix, num_matrix, initial_guess=None):
    # 1) Fit Elo
    row_elos, col_elos, res = compute_elo_ratings_binomial_rect(win_rate_matrix, num_matrix, initial_guess)
    print("Fitted row-players Elo:", row_elos)
    print("Fitted col-players Elo:", col_elos)
    shift = translation(list(row_elos)+list(col_elos))
    print(f'Translating all Domain ELOs by {shift} to match General ELOs')

    # 2) Parametric bootstrap confidence intervals
    (row_mean, row_low, row_high,
     col_mean, col_low, col_high) = bootstrap_elo_confidence_intervals_rect(
         np.array(list(row_elos) + list(col_elos)), num_matrix, initial_guess, n_samples=200, confidence=0.95, random_seed=42
    )
    
    for l in [row_elos, col_elos, row_mean, row_low, row_high, col_mean, col_low, col_high]:
        l += shift
    return row_elos, col_elos, row_mean, row_low, row_high,col_mean, col_low, col_high

def plot_elo(win_rate_matrix, num_matrix, initial_guess=None):
    M = win_rate_matrix.shape[0]
    N = win_rate_matrix.shape[1]

    # 1) Fit Elo
    row_elos, col_elos, row_mean, row_low, row_high,col_mean, col_low, col_high = get_elo_bounds(win_rate_matrix, num_matrix, initial_guess=initial_guess)

    # 3) Simple Plot
    plt.figure(figsize=(8, 4))
    # We'll put row-players on the left, col-players on the right
    x_row = np.arange(M)
    x_col = np.arange(N) + M + 1  # shift them to the right

    plt.bar(x_row, row_elos, color='blue', alpha=0.6, label='Row-players (MLE)')
    plt.errorbar(x_row, row_mean,
                 yerr=[row_mean - row_low, row_high - row_mean],
                 fmt='none', ecolor='blue', capsize=4)

    plt.bar(x_col, col_elos, color='green', alpha=0.6, label='Col-players (MLE)')
    plt.errorbar(x_col, col_mean,
                 yerr=[col_mean - col_low, col_high - col_mean],
                 fmt='none', ecolor='green', capsize=4)

    labels = [f"Row{i}" for i in range(M)] + [f"Col{j}" for j in range(N)]
    plt.xticks(list(x_row) + list(x_col), labels, rotation=45)
    plt.ylabel("Elo rating")
    plt.title("Rectangular Elo Fit (Row vs. Column Players)")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return (row_elos, col_elos, row_mean, row_high, row_low, col_mean, col_high, col_low)

def fit_four_models_and_select(ELO_LIST, col_mean, col_low=None, col_high=None):
    """
    Fit four models to (x,y) data and choose the best-fitting model
    by HIGHEST AIC. (Nonstandard: typically one picks LOWEST AIC.)

    Uses a hard clipping approach for saturation, and incorporates error bars
    from col_low/col_high (if provided) for weighted fitting.
    
    Parameters
    ----------
    ELO_LIST : array-like
        x-values of the data
    col_mean : array-like
        y-values of the data
    col_low, col_high : array-like or None
        Defines the asymmetric error bars for each point, so that
            yerr_minus = (col_mean - col_low)
            yerr_plus  = (col_high - col_mean)
        We'll then take col_err = 0.5 * (yerr_minus + yerr_plus)
        and use weights = 1 / col_err for the fits.
        
    Returns
    -------
    result_dict : dict
        Keys:
            'best_model_name': str
            'best_params': array of best-fit parameters
            'best_func': callable (the best-fit function)
            'all_results': dict with each model's AIC, parameters, SSR
    """
    import numpy as np
    import lmfit
    
    def compute_r2(y_true, y_pred):
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        return 1 - ss_res / ss_tot if ss_tot != 0 else 0

    # --------------------------------------------------------------------------------
    # Define functions with hard clipping (no smoothing parameter)
    # --------------------------------------------------------------------------------
    def linear_func(x, a, b):
        return a * x + b

    def linear_upper_clip_func(x, a, b, x_high):
        # saturates at plateau = a*x_high + b, with a hard clip
        f_lin = a * x + b
        plateau = a * x_high + b
        return np.minimum(f_lin, plateau)

    def linear_lower_clip_func(x, a, b, x_low):
        # saturates at floor_val = a*x_low + b, with a hard clip
        f_lin = a * x + b
        floor_val = a * x_low + b
        return np.maximum(f_lin, floor_val)

    def smooth_clip_func(x, fmin, fmax, x_low, x_high):
        # saturates at fmin (low) and fmax (high) with a hard clip
        x = np.array(x, copy=False)
        f_lin = fmin + (fmax - fmin) * (x - x_low) / (x_high - x_low)
        z = np.maximum(f_lin, fmin)
        z_clamped = np.minimum(z, fmax)
        return z_clamped

    # --------------------------------------------------------------------------------
    # Convert inputs to numpy; optionally compute error bars for weighting
    # --------------------------------------------------------------------------------
    ELO_LIST = np.array(ELO_LIST, dtype=float)
    col_mean = np.array(col_mean, dtype=float)

    if col_low is not None and col_high is not None:
        col_low = np.array(col_low, dtype=float)
        col_high = np.array(col_high, dtype=float)
        yerr_minus = col_mean - col_low
        yerr_plus = col_high - col_mean
        col_err = 0.5 * (yerr_minus + yerr_plus)
        col_err[col_err <= 0] = np.median(col_err[col_err > 0])
        weights = 1.0 / col_err
    else:
        col_err = None
        weights = None

    results = {}

    # --------------------------------------------------------------------------------
    # Helper to make and run each model with initial guesses
    # --------------------------------------------------------------------------------
    def run_fit(model, params, x, y, w=None):
        if w is not None:
            fit_res = model.fit(y, params, x=x, weights=w)
        else:
            fit_res = model.fit(y, params, x=x)
        return fit_res

    # --------------------------------------------------------------------------------
    # a) Pure linear
    # --------------------------------------------------------------------------------
    p0_lin = {
        'a': (max(col_mean) - min(col_mean)) / (max(ELO_LIST) - min(ELO_LIST)),
        'b': np.mean(col_mean) - np.mean(ELO_LIST),
    }
    model_lin = lmfit.Model(linear_func, independent_vars=['x'])
    params_lin = model_lin.make_params(**p0_lin)
    result_lin = run_fit(model_lin, params_lin, ELO_LIST, col_mean, w=weights)
    popt_lin = [result_lin.params['a'].value, result_lin.params['b'].value]
    results["linear"] = {
        "popt": popt_lin,
        "SSR": result_lin.chisqr,
        "AIC": result_lin.aic,
        "func": linear_func
    }

    # --------------------------------------------------------------------------------
    # b) Linear + upper cutoff
    # --------------------------------------------------------------------------------
    init_x_high = np.quantile(ELO_LIST, 0.95)
    p0_up = {
        'a': p0_lin['a'],
        'b': p0_lin['b'],
        'x_high': init_x_high
    }
    model_up = lmfit.Model(linear_upper_clip_func, independent_vars=['x'])
    params_up = model_up.make_params(**p0_up)
    params_up['x_high'].set(min=np.quantile(ELO_LIST, 0.80), max=max(ELO_LIST))
    result_up = run_fit(model_up, params_up, ELO_LIST, col_mean, w=weights)
    popt_up = [
        result_up.params['a'].value,
        result_up.params['b'].value,
        result_up.params['x_high'].value
    ]
    results["linear_upper_cut"] = {
        "popt": popt_up,
        "SSR": result_up.chisqr,
        "AIC": result_up.aic,
        "func": linear_upper_clip_func
    }

    # --------------------------------------------------------------------------------
    # c) Linear + lower cutoff
    # --------------------------------------------------------------------------------
    p0_low = {
        'a': p0_lin['a'],
        'b': p0_lin['b'],
        'x_low': min(ELO_LIST)
    }
    model_low = lmfit.Model(linear_lower_clip_func, independent_vars=['x'])
    params_low = model_low.make_params(**p0_low)
    result_low = run_fit(model_low, params_low, ELO_LIST, col_mean, w=weights)
    popt_low = [
        result_low.params['a'].value,
        result_low.params['b'].value,
        result_low.params['x_low'].value
    ]
    results["linear_lower_cut"] = {
        "popt": popt_low,
        "SSR": result_low.chisqr,
        "AIC": result_low.aic,
        "func": linear_lower_clip_func
    }

    # --------------------------------------------------------------------------------
    # d) Linear with both upper & lower cutoffs
    # --------------------------------------------------------------------------------
    p0_both = {
        'fmin': np.quantile(col_mean, 0.25),
        'fmax': np.quantile(col_mean, 0.75),
        'x_low': np.quantile(ELO_LIST, 0.25),
        'x_high': np.quantile(ELO_LIST, 0.75),
    }
    model_both = lmfit.Model(smooth_clip_func, independent_vars=['x'])
    params_both = model_both.make_params(**p0_both)
    result_both = run_fit(model_both, params_both, ELO_LIST, col_mean, w=weights)
    popt_both = [
        result_both.params['fmin'].value,
        result_both.params['fmax'].value,
        result_both.params['x_low'].value,
        result_both.params['x_high'].value
    ]
    results["linear_both_cut"] = {
        "popt": popt_both,
        "SSR": result_both.chisqr,
        "AIC": result_both.aic,
        "func": smooth_clip_func
    }
    
    for model_key, model_info in results.items():
        # Evaluate the model function on all x-values using the best-fit parameters.
        y_pred = model_info["func"](ELO_LIST, *model_info["popt"])
        model_info["R2"] = compute_r2(col_mean, y_pred)

    # --------------------------------------------------------------------------------
    # Pick best model by AIC (here lower AIC is preferred)
    # --------------------------------------------------------------------------------
    best_model_name = min(results.keys(), key=lambda m: results[m]["AIC"])
    best_model_info = results[best_model_name]

    # Print the best model and its R² value
    print("Best model: {}".format(best_model_name))
    print("R²: {:.4f}".format(best_model_info["R2"]))
    

    return {
        "best_model_name": best_model_name,
        "best_params": best_model_info["popt"],
        "best_func": best_model_info["func"],
        "all_results": results
    }

def get_thisfile_dir():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'elo')

def get_llm_elos(fname='llm_elos.csv'):
    current_dir = get_thisfile_dir()
    elos = pd.read_csv(f'{current_dir}/{fname}')
    return elos


# ------------------------------------------------
# Plot data for col_mean and row_mean + best-fit model
# ------------------------------------------------

# Plot col_mean with error bars
def plot_elo_standard(row_mean, row_low, row_high, col_mean, col_low, col_high,
                      rowlabel='X', collabel='Y', gamename='tictactoe',
                      models_to_ignore=[], points_to_exclude=[], ax=None):
    # Get elo data and filter out unwanted models
    elos = get_llm_elos()
    elos = elos[~elos['llm_name'].isin(models_to_ignore)]
    ELO_LIST = elos['elo'].tolist()
    xerr_neg = elos['xerr_neg'].tolist()
    xerr_pos = elos['xerr_pos'].tolist()
    
    assert len(row_mean) == len(ELO_LIST), "Make sure you're using the right models in the right order. See llm_elos.csv"

    # Determine indices for inliers (used for fitting) and outliers
    fit_indices = [i for i in range(len(ELO_LIST)) if i not in points_to_exclude]
    outlier_indices = [i for i in points_to_exclude if i < len(ELO_LIST)]
    
    # Prepare inlier data for both row and col metrics
    ELO_inlier     = [ELO_LIST[i] for i in fit_indices]
    row_mean_inlier = [row_mean[i] for i in fit_indices]
    row_low_inlier  = [row_low[i] for i in fit_indices]
    row_high_inlier = [row_high[i] for i in fit_indices]
    col_mean_inlier = [col_mean[i] for i in fit_indices]
    col_low_inlier  = [col_low[i] for i in fit_indices]
    col_high_inlier = [col_high[i] for i in fit_indices]
    xerr_neg_inlier = [xerr_neg[i] for i in fit_indices]
    xerr_pos_inlier = [xerr_pos[i] for i in fit_indices]
    
    # Prepare outlier data
    ELO_outlier     = [ELO_LIST[i] for i in outlier_indices]
    row_mean_outlier = [row_mean[i] for i in outlier_indices]
    col_mean_outlier = [col_mean[i] for i in outlier_indices]

    # Create a new figure/axes if no axis is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Plot error bars for inlier points using the provided axis
    ax.errorbar(
        ELO_inlier,
        col_mean_inlier,
        fmt='ro',
        markersize=2,
        yerr=[
            [cm - cl for cm, cl in zip(col_mean_inlier, col_low_inlier)],
            [ch - cm for ch, cm in zip(col_high_inlier, col_mean_inlier)]
        ],
        xerr=[xerr_neg_inlier, xerr_pos_inlier],
        label=collabel
    )
    ax.errorbar(
        ELO_inlier,
        row_mean_inlier,
        fmt='bo',
        markersize=2,
        yerr=[
            [rm - rl for rm, rl in zip(row_mean_inlier, row_low_inlier)],
            [rh - rm for rh, rm in zip(row_high_inlier, row_mean_inlier)]
        ],
        xerr=[xerr_neg_inlier, xerr_pos_inlier],
        label=rowlabel
    )

    # Plot outliers as scatter points so they are still visible on the plot
    if outlier_indices:
        ax.scatter(ELO_outlier, col_mean_outlier, c='r', marker='s',
                   label=f"{collabel} Outlier", zorder=5)
        ax.scatter(ELO_outlier, row_mean_outlier, c='b', marker='s', 
                   label=f"{rowlabel} Outlier", zorder=5)

    # Fit and plot for col_mean using the inlier data
    result = fit_four_models_and_select(ELO_inlier, col_mean_inlier, col_low_inlier, col_high_inlier)
    print("COL MEAN Best Model Name:", result["best_model_name"])
    print("COL MEAN Best-Fit Parameters:", result["best_params"])
    for m_name, info in result["all_results"].items():
        print(f"  {m_name:>18s}  AIC={info['AIC']:.2f},  SSR={info['SSR']:.2f}")

    x_eval = np.linspace(min(ELO_inlier), max(ELO_inlier), 100)
    y_eval = result["best_func"](x_eval, *result["best_params"])
    ax.plot(x_eval, y_eval, 'r-')

    # Fit and plot for row_mean using the inlier data
    result = fit_four_models_and_select(ELO_inlier, row_mean_inlier, row_low_inlier, row_high_inlier)
    print("ROW MEAN Best Model Name:", result["best_model_name"])
    print("ROW MEAN Best-Fit Parameters:", result["best_params"])
    for m_name, info in result["all_results"].items():
        print(f"  {m_name:>18s}  AIC={info['AIC']:.2f},  SSR={info['SSR']:.2f}")

    x_eval = np.linspace(min(ELO_inlier), max(ELO_inlier), 100)
    y_eval = result["best_func"](x_eval, *result["best_params"])
    ax.plot(x_eval, y_eval, 'b-')
    
    ax.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='y', nbins=5)
    ax.set_ylim(800, 1650)
    ax.set_yticks([1000, 1200, 1400, 1600])
    ax.set_xlim(1050, 1450)
    ax.set_xticks([1100, 1200, 1300, 1400])

    ax.set_xlabel("General Elo")
    ax.set_ylabel("Domain Elo") 
#    ax.legend()
    ax.set_title(f"{gamename}")

    # Save the figure using the axis's parent figure (if desired)
    d = get_thisfile_dir()
    os.makedirs(f'{d}/figures/', exist_ok=True)
    ax.figure.savefig(f'{d}/figures/{gamename}_elo.pdf', bbox_inches='tight')
    ax.figure.savefig(f'{d}/figures/{gamename}_elo.png', bbox_inches='tight', dpi=300)

    # Only show if we're not already in a subplot framework (optional)
    if ax is None:
        plt.show()

# %%
def plot_win_matrix(WIN_RATE_MATRIX, rowlabel = 'O', collabel = 'X', gamename = 'tictactoe', models_to_ignore = []):
    """Plot win rate matrix with model names from csv file.
    
    Args:
        WIN_RATE_MATRIX: 2D numpy array of win rates
        csv_path: Path to CSV file containing model names. Defaults to 'llm_elos.csv'
                 in same directory as calling script.
    """
    # Get directory of csv relative to calling script
    elos = get_llm_elos()
    MODEL_LIST = [model_name for model_name in elos['llm_name'] if model_name not in models_to_ignore]
    MODEL_LIST = [model_name.split("/")[-1] for model_name in MODEL_LIST]
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 14,
    })
    # Flip the matrix horizontally since we want increasing ELO from left to right
    plt.imshow(np.fliplr(WIN_RATE_MATRIX), cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.xlabel(collabel)
    plt.ylabel(rowlabel)
    plt.xticks(ticks=range(len(MODEL_LIST)), labels=MODEL_LIST[::-1], rotation=90)
    plt.yticks(ticks=range(len(MODEL_LIST)), labels=MODEL_LIST)
    plt.title(f'{gamename}: Win rate of {rowlabel}')
    
    # Add arrow for increasing ELO
    plt.annotate('', xy=(len(MODEL_LIST)-1, len(MODEL_LIST)-1), 
                xytext=(0, len(MODEL_LIST)-1),
                arrowprops=dict(facecolor='black', shrink=0.05),
                ha='right', va='center')
    plt.annotate('', xy=(0, 0), 
                xytext=(0, len(MODEL_LIST)-1),
                arrowprops=dict(facecolor='black', shrink=0.05),
                ha='right', va='center')
    
    d = get_thisfile_dir()
    os.makedirs(f'{d}/figures/', exist_ok = True)
    plt.savefig(f'{d}/figures/{gamename}_winrate.pdf', bbox_inches='tight')
    plt.savefig(f'{d}/figures/{gamename}_winrate.png', bbox_inches='tight', dpi = 300)

# %%

def gather_game_specific_elos(game_dict):
    """
    game_dict is an OrderedDict or list of tuples of the form:
      {
        'gamename1': (win_rate_matrix_1, num_matrix_1),
        'gamename2': (win_rate_matrix_2, num_matrix_2),
        ...
      }
    or a list like
      [
        ('gamename1', win_rate_matrix_1, num_matrix_1),
        ('gamename2', win_rate_matrix_2, num_matrix_2),
      ]
    
    Returns
    -------
    E : np.ndarray of shape (G, P)
        E[g, m] = Elo rating of “player slot” m in game g.
                  The row players come first, then the column players.
    game_names : list of length G
    row_col_sizes : list of length G, where each entry = (M_g, N_g)
                    So you know how to map which portion is row or column for each game.
    """
    game_names = []
    E_list = []
    row_col_sizes = []
    
    # If the dictionary is not truly "ordered", might convert to list of items
    # but let's assume we have a list or an OrderedDict for reproducible order
    if isinstance(game_dict, dict):
        items = list(game_dict.items())
    else:
        # Already a list
        items = game_dict
    
    for idx, entry in enumerate(items):
        if isinstance(entry, tuple) and len(entry) == 3:
            gamename, win_mat, num_mat = entry
        else:
            # e.g. 'gamename': (win_mat, num_mat)
            gamename, (win_mat, num_mat) = entry
        
        game_names.append(gamename)
        
        # Fit rectangular Elo for that game:
        row_elos, col_elos, row_mean, row_low, row_high,col_mean, col_low, col_high = get_elo_bounds(win_mat, num_mat, initial_guess= list(ELO_LIST) + list(ELO_LIST))
        
        # Store in E vector for that game:
        # We treat these row/col players as distinct. So length = M + N.
        this_game_E = np.concatenate([row_elos, col_elos])
        E_list.append(this_game_E)
        
        row_col_sizes.append((win_mat.shape[0], win_mat.shape[1]))
        
    # Stack into shape (G, P)
    E = np.vstack(E_list)
    return E, game_names, row_col_sizes

def method1_average_elo(E):
    """
    E has shape (G, P), 
    where G = #games, P = #player-slots per game (or total if each game is consistent).
    Returns a length-P vector of the 'global' rating via simple average.
    """
    return np.mean(E, axis=0)

def method2_aggregate_and_fit(game_dict):
    """
    1) For each game in game_dict, assumes the win rate matrix (wmat) is of shape
       (num_models, num_models).  Aggregates the wins and plays over all games.
    2) For each cell (i,j), total wins = sum_game [wmat[i,j] * num_matrix[i,j]]
       and total plays = sum_game [num_matrix[i,j]].
    3) Compute the aggregated win rate as total_wins/total_plays.
    4) Fit Elo on the aggregated win rate matrix using compute_elo_ratings_binomial_rect.
    5) Return a global rating vector computed as the average of row and column Elo.
    """
    import numpy as np

    # Parse the game list.
    if isinstance(game_dict, dict):
        items = list(game_dict.items())
    else:
        items = game_dict

    # Use the first game to define the shape.
    first_entry = items[0]
    if len(first_entry) == 3:
        _, wmat, nmat = first_entry
    else:
        _, (wmat, nmat) = first_entry

    num_models_row = wmat.shape[0]
    num_models_col = wmat.shape[1]
    # We require that the same set of models appear as both row and column players.
    assert num_models_row == num_models_col, "Each win matrix must be square (same number of row and col models)."
    num_models = num_models_row

    # Initialize aggregated wins and plays matrices.
    agg_wins = np.zeros((num_models, num_models), dtype=float)
    agg_plays = np.zeros((num_models, num_models), dtype=float)

    # Sum over games.
    for entry in items:
        if len(entry) == 3:
            _, wmat, nmat = entry
        else:
            _, (wmat, nmat) = entry
        for i in range(num_models):
            for j in range(num_models):
                if nmat[i, j] is None or nmat[i, j] == 0:
                    continue
                agg_plays[i, j] += nmat[i, j]
                agg_wins[i, j] += wmat[i, j] * nmat[i, j]

    eps = 1e-12
    agg_fraction = agg_wins / agg_plays

    print("Aggregated matrix shape:", agg_fraction.shape, agg_plays.shape)

    # Fit Elo using the aggregated matrix.
    # get_elo_bounds returns:
    #   row_elos, col_elos, row_mean, row_low, row_high, col_mean, col_low, col_high
    row_elos, col_elos, _, _, _, _, _, _ = get_elo_bounds(agg_fraction, agg_plays, initial_guess=list(ELO_LIST) + list(ELO_LIST))


    return np.array(list(row_elos) + list(col_elos))


def method3_pca_correlation(E):
    """
    E has shape (G, P). We'll compute the correlation across the G 'rows'.
    i.e. each row E[g,:] is a vector in R^P. 
    The correlation matrix is GxG, measuring correlation between E[g,:] and E[h,:].
    
    Then pick the largest-eigenvalue eigenvector u in R^G.
    We'll L2-normalize u.  The global rating for each player-slot m is sum_g u[g]*E[g,m].
    
    Returns
    -------
    global_rating_3 : length-P array of the player's “Method-3 global rating”
    u               : length-G array, the principal eigenvector
    """
    G, P = E.shape
    
    # Compute correlation across rows
    # E[g,:], E[h,:]
    corr_mat = np.corrcoef(E)  # shape (G, G)
    print(corr_mat)
    
    # Eigen-decomposition
    vals, vecs = np.linalg.eig(corr_mat)
    # Identify principal (largest) eigenvalue
    idx = np.argmax(vals.real)
    u = vecs[:, idx].real
    # (Sometimes you might want to ensure it's "positively oriented," e.g. by sign.)
    
    # L2-normalize
    norm_u = np.sum(u)
    if norm_u < 1e-15:
        # fallback
        u = np.ones(G, dtype=float)/np.sqrt(G)
    else:
        u /= norm_u
    
    print("Vector u:", u, "Eigenvalue:", vals[idx].real)
    # Now compute global rating
    # global_rating_3[m] = sum_g u[g] * E[g,m]
    global_3 = E.T.dot(u)  # shape (P,)
    
    return global_3, u

def demo_global_rating_plot(game_dict, leave_one_out=True):
    """
    Produces a 3 x G subplot comparing each game's per-player rating (y-axis)
    to the global rating (x-axis) computed by each method.
    In addition to the scatter plot, this modified version performs a fit 
    (separately for Houdini and Guard) in the same style as plot_elo_standard.
    
    Parameters:
      game_dict: a dict or list of tuples, where each entry is:
                 ('gamename', win_rate_matrix, num_matrix)
      leave_one_out: bool, if True perform leave-one-out global rating computation.
    """
    # 1) Gather each game's Elo ratings (from a per-game fit)
    E_game, game_names, row_col_sizes = gather_game_specific_elos(game_dict)
    G, P = E_game.shape

    # 2) Compute leave-one-out global ratings (or use all-games global ratings)
    if leave_one_out:
        E_global_1_loo = np.zeros((G, P))
        E_global_2_loo = np.zeros((G, P))
        E_global_3_loo = np.zeros((G, P))
        
        for i in range(G):
            # Method 1: Average Elo leaving out game i.
            E_global_1_loo[i, :] = np.mean(np.delete(E_game, i, axis=0), axis=0)
            
            # Method 2: Block aggregation leaving out game i.
            if isinstance(game_dict, dict):
                reduced_dict = {k: v for k, v in game_dict.items() if k != game_names[i]}
            else:
                reduced_dict = [entry for j, entry in enumerate(game_dict) if j != i]
            E2_block_reduced = method2_aggregate_and_fit(reduced_dict)
            E_global_2_loo[i, :] = E2_block_reduced
            
            # Method 3: PCA leaving out game i.
            E_game_reduced = np.delete(E_game, i, axis=0)
            E_global_3_loo[i, :], _ = method3_pca_correlation(E_game_reduced)
        
        method_globals = [
            (E_global_1_loo, "Method 1 LOO"),
            (E_global_2_loo, "Method 2 LOO"),
            (E_global_3_loo, "Method 3 (PCA) LOO"),
        ]
    else:
        E_global_1 = method1_average_elo(E_game)
        E2_block = method2_aggregate_and_fit(game_dict)
        E_global_2 = np.mean(E2_block, axis=0)
        E_global_3, _ = method3_pca_correlation(E_game)
        method_globals = [
            (E_global_1, "Method 1"),
            (E_global_2, "Method 2"),
            (E_global_3, "Method 3 (PCA)"),
        ]
    
    # For re-running game-specific fits, get the win_rate and game count matrices.
    # Convert game_dict to a list of tuples: (gamename, win_rate_matrix, num_matrix)
    if isinstance(game_dict, dict):
        game_items = [(k, v[0], v[1]) for k, v in game_dict.items()]
    else:
        game_items = [(entry[0], entry[1], entry[2]) for entry in game_dict]
    
    
    # 3) Create a 3 x G subplot grid.
    fig, axes = plt.subplots(nrows=3, ncols=G, figsize=(4 * G, 9), sharey=False)
    if G == 1:
        axes = np.array([axes]).T

    for j, (global_mat, method_name) in enumerate(method_globals):
        for i in range(G):
            ax = axes[j, i]
            num_models = len(ELO_LIST)  # assuming both groups have the same length as CSV
            
            # Scatter the leave-one-out global ratings vs. game-specific Elo.
            # Here we assume that in the concatenated vectors:
            # first half corresponds to Houdini (row players) and second half to Guard (column players)
            ax.scatter(global_mat[i, :num_models], E_game[i, :num_models],
                       c='r', label='Houdini')
            ax.scatter(global_mat[i, num_models:], E_game[i, num_models:],
                       c='b', label='Guard')
            
            # Retrieve the win_rate_matrix and num_matrix for the current game.
            gamename = game_names[i]
            win_rate_matrix, num_matrix = None, None
            for item in game_items:
                if item[0] == gamename:
                    win_rate_matrix, num_matrix = item[1], item[2]
                    break
            if win_rate_matrix is None:
                continue  # skip if not found
            
            # Re-run the game-specific Elo fitting to obtain error bars and mean ratings.
            row_elos, col_elos, row_mean, row_low, row_high, col_mean, col_low, col_high = get_elo_bounds(
                win_rate_matrix, num_matrix, initial_guess=list(ELO_LIST) + list(ELO_LIST)
            )
            
            # --- Fitting Houdini (row players) ---
            fit_houdini = fit_four_models_and_select(global_mat[i, :num_models], row_mean, row_low, row_high)
            x_eval = np.linspace(min(global_mat[i, :num_models]), max(global_mat[i, :num_models]), 100)
            y_fit_houdini = fit_houdini["best_func"](x_eval, *fit_houdini["best_params"])
            corr_houdini = np.corrcoef(global_mat[i, :num_models], row_mean)[0, 1]
            
            # --- Fitting Guard (column players) ---
            fit_guard = fit_four_models_and_select(global_mat[i, num_models:], col_mean, col_low, col_high)
            x_eval_guard = np.linspace(min(global_mat[i, num_models:]), max(global_mat[i, num_models:]), 100)
            y_fit_guard = fit_guard["best_func"](x_eval_guard, *fit_guard["best_params"])
            corr_guard = np.corrcoef(ELO_LIST, col_mean)[0, 1]
            
            # Overlay the fitted curves.
            ax.plot(x_eval, y_fit_houdini, 'r--')
            ax.plot(x_eval_guard, y_fit_guard, 'b--')
            
            ax.set_title(f"Game {i+1} vs {method_name}")
            ax.set_xlabel("Global rating (CSV)")
            ax.set_ylabel("Game-specific Elo")
            ax.legend()
    
    plt.tight_layout()
    plt.show()
    
def final_global_rating_plot(game_dict, leave_one_out=True):
    """
    Produces a 3 x G subplot comparing each game's per-player rating (y-axis)
    to the global rating (x-axis) computed by each method.
    In addition to the scatter plot, this modified version performs a fit 
    (separately for Houdini and Guard) in the same style as plot_elo_standard.
    
    Parameters:
      game_dict: a dict or list of tuples, where each entry is:
                 ('gamename', win_rate_matrix, num_matrix)
      leave_one_out: bool, if True perform leave-one-out global rating computation.
    """
    # 1) Gather each game's Elo ratings (from a per-game fit)
    E_game, game_names, row_col_sizes = gather_game_specific_elos(game_dict)
    G, P = E_game.shape
    
    if isinstance(game_dict, dict):
        game_items = [(k, v[0], v[1]) for k, v in game_dict.items()]
    else:
        game_items = [(entry[0], entry[1], entry[2]) for entry in game_dict]
    
    global_mat = np.zeros((G, P))
    
    if leave_one_out:
        for i in range(G):
            # Method 1: Average Elo leaving out game i.
            global_mat[i, :] = np.mean(np.delete(E_game, i, axis=0), axis=0)
    else:
        for i in range(G):
            # Method 1: Average Elo leaving out game i.
            global_mat[i, :] = np.mean(E_game, axis=0)

    fig, axes = plt.subplots(nrows=1, ncols=G, figsize=(4 * G, 3), sharey=False)

    for i in range(G):
        ax = axes[i]
        num_models = len(ELO_LIST)  # assuming both groups have the same length as CSV
        
        # Scatter the leave-one-out global ratings vs. game-specific Elo.
        # Here we assume that in the concatenated vectors:
        # first half corresponds to Houdini (row players) and second half to Guard (column players)
        ax.scatter(global_mat[i, :num_models], E_game[i, :num_models],
                    c='r', label='Houdini')
        ax.scatter(global_mat[i, num_models:], E_game[i, num_models:],
                    c='b', label='Guard')
        
        # Retrieve the win_rate_matrix and num_matrix for the current game.
        gamename = game_names[i]
        win_rate_matrix, num_matrix = None, None
        for item in game_items:
            if item[0] == gamename:
                win_rate_matrix, num_matrix = item[1], item[2]
                break
        if win_rate_matrix is None:
            continue  # skip if not found
        
        # Re-run the game-specific Elo fitting to obtain error bars and mean ratings.
        row_elos, col_elos, row_mean, row_low, row_high, col_mean, col_low, col_high = get_elo_bounds(
            win_rate_matrix, num_matrix, initial_guess=list(ELO_LIST) + list(ELO_LIST)
        )
        
        # --- Fitting Houdini (row players) ---
        fit_houdini = fit_four_models_and_select(global_mat[i, :num_models], row_mean, row_low, row_high)
        x_eval = np.linspace(min(global_mat[i, :num_models]), max(global_mat[i, :num_models]), 100)
        y_fit_houdini = fit_houdini["best_func"](x_eval, *fit_houdini["best_params"])
        corr_houdini = np.corrcoef(global_mat[i, :num_models], row_mean)[0, 1]
        
        # --- Fitting Guard (column players) ---
        fit_guard = fit_four_models_and_select(global_mat[i, num_models:], col_mean, col_low, col_high)
        x_eval_guard = np.linspace(min(global_mat[i, num_models:]), max(global_mat[i, num_models:]), 100)
        y_fit_guard = fit_guard["best_func"](x_eval_guard, *fit_guard["best_params"])
        corr_guard = np.corrcoef(ELO_LIST, col_mean)[0, 1]
        
        # Overlay the fitted curves.
        ax.plot(x_eval, y_fit_houdini, 'r--')
        ax.plot(x_eval_guard, y_fit_guard, 'b--')
        
        ax.set_xlabel("Global rating (CSV)")
        ax.set_ylabel("Game-specific Elo")
        ax.legend()

    plt.tight_layout()
    plt.show()


# %%
'''
# MINIMAL WORKING EXAMPLE
row_elos, col_elos, row_mean, row_low, row_high,col_mean, col_low, col_high = get_elo_bounds(win_rate, num_played)
# ^ this takes 75s for me to run
# NOTE MAKE SURE WIN_RATE IS GUARD WIN RATE. Rows should be Guard, Columns should be Houdini. Value should be guard win rate
plot_elo_standard(row_mean, row_low, row_high, col_mean, col_low, col_high, collabel = 'Guard', rowlabel = 'Houdini', gamename = 'backdoorcode')
plot_win_matrix(1-win_rate, rowlabel = 'Houdini', collabel = 'Guard', gamename = 'backdoorcode')

'''



# %%
