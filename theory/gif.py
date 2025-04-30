# %%
"""zigzag_grid.py

Animate multiple Guard↔Houdini zig‑zag strategies side‑by‑side.

Usage
-----
from zigzag_grid import zigzag_bounce_grid

param_sets = [
    dict(monitor_slope=1, ai_slope=2, ai_intercept=-800, ai_g_elo=1000),
    dict(monitor_slope=1, ai_slope=1, ai_intercept=400,  ai_g_elo=1200),
    dict(monitor_slope=1, ai_slope=3, ai_intercept=-2000, ai_g_elo=1000),
]

ani = zigzag_bounce_grid(param_sets, fps=24, save_path='grid.gif')
ani  # displays inline in Jupyter
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from decimal import Decimal, getcontext

# high‑precision maths so the win‑probability is rock‑solid
getcontext().prec = 60


# ───────────────────────── generic helpers ───────────────────────────────────

def _p_a_beats_b(elo_a: Decimal, elo_b: Decimal) -> Decimal:
    """Probability that player A beats player B under Elo."""
    return Decimal(1) / (Decimal(1) + Decimal(10) ** ((elo_b - elo_a) / Decimal(400)))


def _overall_win_prob(
    monitor_slope: float,
    ai_slope: float,
    ai_intercept: float,
    ai_g_elo: float,
    n_steps: int,
) -> Decimal:
    """Overall probability the Guard wins a *n_steps*-game match."""
    g_grid = np.linspace(0, ai_g_elo, n_steps + 1, dtype=float)
    g_dec  = [Decimal(str(x)) for x in g_grid]

    m_elos = [Decimal(str(monitor_slope)) * g for g in g_dec[:-1]]
    a_elos = [
        Decimal(str(ai_slope)) * g + Decimal(str(ai_intercept))
        for g in g_dec[1:]
    ]

    result = Decimal(1)
    for m, a in zip(m_elos, a_elos):
        result *= _p_a_beats_b(m, a)
    return result


def _best_num_steps(
    monitor_slope: float,
    ai_slope: float,
    ai_intercept: float,
    ai_g_elo: float,
    max_steps: int = 20,
) -> int:
    """Return n^* that maximises the Guard's win probability."""
    best_n, best_p = 1, Decimal(0)
    for n in range(1, max_steps + 1):
        p = _overall_win_prob(monitor_slope, ai_slope, ai_intercept, ai_g_elo, n)
        if p > best_p:
            best_n, best_p = n, p
    return best_n


def _zigzag_vertices(
    monitor_slope: float,
    ai_slope: float,
    ai_intercept: float,
    ai_g_elo: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return x and y coordinates of the optimal zig‑zag path."""
    n = _best_num_steps(monitor_slope, ai_slope, ai_intercept, ai_g_elo)

    g   = np.linspace(0, ai_g_elo, n + 1)
    m_y = monitor_slope * g
    a_y = ai_slope * g + ai_intercept

    xs, ys = [0.0], [0.0]   # start at origin
    for i in range(n):
        xs.append(g[i + 1]); ys.append(a_y[i + 1])       # game step (diag)
        if i + 1 < n:
            xs.append(g[i + 1]); ys.append(m_y[i + 1])   # transfer step (vert)
    return np.array(xs), np.array(ys)


# ──────────────────────── animation factory ──────────────────────────────────
def zigzag_bounce_grid(
    param_sets: list[dict],
    *,
    fps: int,
    save_path: str,
    duration: float,
    font_size: float,
    line_width: float,
):
    """Render *k* zig‑zag strategies in a single animated GIF/HTML5 video.

    Parameters
    ----------
    param_sets
        List of dictionaries with keys:
            monitor_slope, ai_slope, ai_intercept, ai_g_elo
    fps
        Frames per second.
    save_path
        If given, write the GIF to disk.
    duration
        Total runtime *in seconds* for the animation. All panels finish
        together, no matter how many segments each one needs.
    font_size
        Base font size for text elements in the plot.
    line_width
        Width of the lines in the plot.
    """

    n_plots = len(param_sets)
    if n_plots == 0:
        raise ValueError("param_sets must contain at least one dictionary.")

    total_frames = max(1, int(duration * fps))

    # Set font sizes based on the provided font_size parameter
    plt.rcParams.update({
        'font.size': font_size,
        'axes.labelsize': font_size * 1.1,
        'axes.titlesize': font_size * 1.1,
        'xtick.labelsize': font_size * 0.9,
        'ytick.labelsize': font_size * 0.9,
        'legend.fontsize': font_size * 0.9,
    })

    # ───── pre‑compute curves + vertices for every subplot ───────────────────
    data = []   # per‑subplot stash
    for p in param_sets:
        xs_curve = np.linspace(0, p['ai_g_elo'], 1000)
        guard_y  = p['monitor_slope'] * xs_curve
        houdini_y = p['ai_slope'] * xs_curve + p['ai_intercept']

        zz_x, zz_y = _zigzag_vertices(**p)
        n_segments = len(zz_x) - 1

        data.append(dict(xs_curve=xs_curve,
                         guard_y=guard_y,
                         houdini_y=houdini_y,
                         zz_x=zz_x,
                         zz_y=zz_y,
                         n_segments=n_segments,
                         p=p))

    # ─────────────── figure + static elements per axis ───────────────────────
    fig, axs = plt.subplots(1, n_plots,
                            figsize=(10, 4),
                            dpi=110,
                            squeeze=False)
    axs = axs[0]

    diag_lines, vert_lines = [], []
    for ax, d in zip(axs, data):
        ax.plot(d['xs_curve'], d['guard_y'],  lw=line_width, color="#3366ff", label="Guard")
        ax.plot(d['xs_curve'], d['houdini_y'], lw=line_width, color="#ff6666", label="Houdini")

        (diag_line,) = ax.plot([], [], lw=line_width, color="#33cc33", linestyle="--", label="Game Step")
        (vert_line,) = ax.plot([], [], lw=line_width, color="#9933ff", linestyle=":", label="Transfer Step")
        
        # Add start and end stars with larger size
        ax.plot(0, 0, '*', color='#3366ff', markersize=20, label='Initial Guard Model')
        ax.plot(d['p']['ai_g_elo'], d['p']['ai_slope'] * d['p']['ai_g_elo'] + d['p']['ai_intercept'], 
                '*', color='#ff6666', markersize=20, label='Target Houdini Model')

        ax.set_xlabel("General Elo")
        ax.set_xlim(0, d['p']['ai_g_elo'])
        min_y = min(d['guard_y'].min(), d['houdini_y'].min())
        max_y = max(d['guard_y'].max(), d['houdini_y'].max())
        
        # Add extra space to the x and y axis
        x_range = d['p']['ai_g_elo']
        y_range = max_y - min_y
        extra_space = 0.1
        ax.set_xlim(-extra_space * x_range, (1 + extra_space) * d['p']['ai_g_elo'])
        ax.set_ylim(min_y - extra_space * y_range, max_y + extra_space * y_range)
        
        if ax is axs[0]:
            ax.set_ylabel("Domain Elo")
        # ax.set_title(
        #     rf"$n^*={_best_num_steps(**d['p'])}$\\n"
        #     rf"m={d['p']['monitor_slope']}, "
        #     rf"a={d['p']['ai_slope']}, "
        #     rf"b={d['p']['ai_intercept']}, "
        #     rf"G={d['p']['ai_g_elo']}"
        # )
        best_num_steps = _best_num_steps(**d['p'])
        ax.set_title(f"Optimal # steps = {best_num_steps}")
        ax.grid(alpha=0.25)
        diag_lines.append(diag_line)
        vert_lines.append(vert_line)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1))

    wspace = 0.37
    left_margin = 0.1
    right_margin = 0.03
    top_margin = 0.27
    bottom_margin = 0.16
    fig.subplots_adjust(left=left_margin, right=1 - right_margin,
                        top=1 - top_margin, bottom=bottom_margin,
                        wspace=wspace)


    # ─────────────────────────── animation update ────────────────────────────
    def update(frame: int):
        t = frame / (total_frames - 1)  # progress 0 → 1
        artists = []
        
        for diag_line, vert_line, d in zip(diag_lines, vert_lines, data):
            seg_float = t * d['n_segments']
            seg_idx = int(np.floor(seg_float))
            seg_idx = min(seg_idx, d['n_segments'] - 1)
            local_t = seg_float - seg_idx

            zz_x, zz_y = d['zz_x'], d['zz_y']

            # vertices fully completed
            full_x = list(zz_x[:seg_idx + 1])
            full_y = list(zz_y[:seg_idx + 1])

            # interpolate along current segment
            x0, y0 = zz_x[seg_idx], zz_y[seg_idx]
            x1, y1 = zz_x[seg_idx + 1], zz_y[seg_idx + 1]
            full_x.append(x0 + local_t * (x1 - x0))
            full_y.append(y0 + local_t * (y1 - y0))

            # bucket into diagonal vs vertical for colouring
            diag_x, diag_y, vert_x, vert_y = [], [], [], []
            for i in range(len(full_x) - 1):
                x_a, x_b = full_x[i], full_x[i + 1]
                y_a, y_b = full_y[i], full_y[i + 1]
                bucket_x, bucket_y = (vert_x, vert_y) if np.isclose(x_a, x_b) else (diag_x, diag_y)
                bucket_x.extend([x_a, x_b, np.nan])
                bucket_y.extend([y_a, y_b, np.nan])

            diag_line.set_data(diag_x, diag_y)
            vert_line.set_data(vert_x, vert_y)
            artists.extend([diag_line, vert_line])
        return artists

    ani = FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=1000 / fps,
        blit=True,
        repeat=True,
    )

    if save_path:
        ani.save(save_path, writer=PillowWriter(fps=fps))
        print(f"GIF saved to {save_path}")

    return ani

# (1, 1, 400, 1200),
# (1, 1, -200, 500),
# (1, 2, -800, 1000),
# (1, 3, -2000, 1000)
param_sets = [
    dict(monitor_slope=1, ai_slope=1, ai_intercept=400,  ai_g_elo=1200),
    dict(monitor_slope=1, ai_slope=1, ai_intercept=-200,  ai_g_elo=500),
    dict(monitor_slope=1, ai_slope=2, ai_intercept=-800, ai_g_elo=1000),
    dict(monitor_slope=1, ai_slope=3, ai_intercept=-2000, ai_g_elo=1000),
]

ani = zigzag_bounce_grid(
    param_sets,
    fps=30,
    duration=4,      # 6-second run, universally applied
    save_path="grid.gif",
    font_size=13,    # control text size
    line_width=2.5,  # control line thickness
)
# %%
