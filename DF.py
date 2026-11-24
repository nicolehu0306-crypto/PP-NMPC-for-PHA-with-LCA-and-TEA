PHA fermentation model — parameter optimisation with differential evolution.

Supports BATCH and FED-BATCH operation.

    F_S   — substrate feed solution flow rate      (L/h)
    F_N   — nitrogen  feed solution flow rate      (L/h)

    F     — NET rate of broth volume change        (L/h)

              F = F_S*(rho_FS - S_F)/rho_w  +  F_N*(rho_FN - N_F)/rho_w

            The (rho_F - C_F)/rho_w factors account for non-ideal volume
            additivity: a concentrated feed solution does not contribute its
            full volume as broth, because the dissolved solute displaces water.

    D     = F / V   — dilution rate                (1/h)

    Batch is the special case F_S = F_N = 0  ->  F = 0  ->  D = 0.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import warnings, json, os
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
#  1. USER SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
EXCEL_PATH  = ''
SHEET_NAME  = 'Sheet1'
HEADER_ROW  = 0
DATASET_IDS = ['DS_original']
OUTPUT_DIR  = ''
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── HOW THE MODE IS CHOSEN ────────────────────────────────────────────────────
#   'interactive' — the script asks you at the terminal (menu + feed settings)
#   'config'      — the script uses DATASET_MODE / DEFAULT_MODE / FEED_CONFIG below
#                   (use this for unattended or batch runs)
MODE_SELECTION = 'interactive'

# ── AVAILABLE FERMENTATION MODES ──────────────────────────────────────────────
MODES = {
    '1': ('batch',
          'Batch — F_S = F_N = 0, so F = 0, D = 0 and V stays constant.'),
    '2': ('fedbatch_constant',
          'Fed-batch, constant feed — fixed F_S and F_N from t_feed_start.'),
    '3': ('fedbatch_exponential',
          'Fed-batch, exponential feed — F(t) = F0*exp(mu_set*t), split by feed_ratio.'),
    '4': ('fedbatch_setpoint',
          'Fed-batch, setpoint control — feed solved on-line to hold S and N constant.'),
}

# Used when MODE_SELECTION = 'config'
DEFAULT_MODE = 'batch'
DATASET_MODE = {
    'DS05': 'batch',
    'DS11': 'batch',
}

# ── FEED CONFIGURATION ────────────────────────────────────────────────────────
# Defaults. In interactive mode you are prompted for the fields that matter for
# the mode you picked; press Enter at any prompt to keep the value shown here.
FEED_CONFIG = {
    # ---- shared by all fed-batch modes ----
    'V0':            1.0,      # initial broth volume                    (L)
    'S_F':          62.93,     # substrate concentration in feed         (g/L)
    'N_F':         164.0,      # nitrogen  concentration in feed         (g/L)
    't_feed_start':  0.0,      # time feeding begins                     (h)

    # ---- densities, used in the volume balance ----
    'rho_FS':    1230.0,       # density of substrate feed solution      (g/L)
    'rho_FN':    1040.0,       # density of nitrogen  feed solution      (g/L)
    'rho_w':     1000.0,       # density of water / broth                (g/L)
    'D_max':        0.30,      # safety cap on dilution rate             (1/h)

    # ---- 'fedbatch_constant' ----
    'F_S_const':    0.010,     # substrate feed flow rate                (L/h)
    'F_N_const':    0.002,     # nitrogen  feed flow rate                (L/h)

    # ---- 'fedbatch_exponential' ----
    'F0':           0.005,     # feed rate at t_feed_start               (L/h)
    'mu_set':       0.15,      # target specific growth rate             (1/h)
    'feed_ratio':   0.85,      # fraction of total feed that is substrate (-)

    # ---- 'fedbatch_setpoint' ----
    'S_sp':         5.0,       # substrate setpoint to maintain          (g/L)
    'N_sp':         0.64,      # nitrogen  setpoint to maintain          (g/L)
}

# Which FEED_CONFIG keys to ask about for each mode
MODE_PROMPTS = {
    'batch': [],
    'fedbatch_constant':    ['V0', 't_feed_start', 'S_F', 'N_F',
                             'F_S_const', 'F_N_const',
                             'rho_FS', 'rho_FN', 'rho_w'],
    'fedbatch_exponential': ['V0', 't_feed_start', 'S_F', 'N_F',
                             'F0', 'mu_set', 'feed_ratio',
                             'rho_FS', 'rho_FN', 'rho_w', 'D_max'],
    'fedbatch_setpoint':    ['V0', 't_feed_start', 'S_F', 'N_F',
                             'S_sp', 'N_sp',
                             'rho_FS', 'rho_FN', 'rho_w', 'D_max'],
}

FIELD_HELP = {
    'V0':           'initial broth volume (L)',
    't_feed_start': 'time feeding begins (h)',
    'S_F':          'substrate concentration in feed (g/L)',
    'N_F':          'nitrogen concentration in feed (g/L)',
    'F_S_const':    'substrate feed flow rate F_S (L/h)',
    'F_N_const':    'nitrogen feed flow rate F_N (L/h)',
    'F0':           'initial total feed rate F0 (L/h)',
    'mu_set':       'target specific growth rate (1/h)',
    'feed_ratio':   'fraction of total feed that is substrate (0-1)',
    'S_sp':         'substrate setpoint to hold (g/L)',
    'N_sp':         'nitrogen setpoint to hold (g/L)',
    'rho_FS':       'density of substrate feed solution (g/L)',
    'rho_FN':       'density of nitrogen feed solution (g/L)',
    'rho_w':        'density of water / broth (g/L)',
    'D_max':        'safety cap on dilution rate (1/h)',
}

# ══════════════════════════════════════════════════════════════════════════════
#  1b. INTERACTIVE MODE SELECTION
# ══════════════════════════════════════════════════════════════════════════════
def ask_float(label, default):
    """Prompt for a number; Enter keeps the default."""
    while True:
        raw = input(f"    {label} [{default}]: ").strip()
        if raw == '':
            return default
        try:
            return float(raw)
        except ValueError:
            print("      Not a number — try again.")


def choose_mode():
    """Show the menu, return the chosen mode string."""
    print("\n" + "="*66)
    print("  FERMENTATION MODE")
    print("="*66)
    for key, (name, desc) in MODES.items():
        print(f"  [{key}] {name}")
        print(f"      {desc}")
    while True:
        choice = input("\n  Select mode [1]: ").strip() or '1'
        if choice in MODES:
            return MODES[choice][0]
        print("  Invalid choice — enter 1, 2, 3 or 4.")


def configure_feed(mode, cfg):
    """Prompt for the feed settings that matter for this mode."""
    fields = MODE_PROMPTS[mode]
    if not fields:
        print(f"\n  Mode '{mode}' needs no feed settings (F = 0, D = 0).")
        return cfg
    print(f"\n  Feed settings for '{mode}'  (Enter = keep default)")
    for key in fields:
        cfg[key] = ask_float(f"{key:14s} {FIELD_HELP[key]:<45s}", cfg[key])
    # Report the resulting volume-correction factors
    fS = (cfg['rho_FS'] - cfg['S_F']) / cfg['rho_w']
    fN = (cfg['rho_FN'] - cfg['N_F']) / cfg['rho_w']
    print(f"\n    Volume correction:  F = F_S*{fS:.5f} + F_N*{fN:.5f}")
    return cfg


def resolve_modes(dataset_ids, cfg):
    """
    Return {dataset_id: mode}. In interactive mode you may apply one mode to
    every dataset, or pick a mode per dataset.
    """
    if MODE_SELECTION != 'interactive':
        return {ds: DATASET_MODE.get(ds, DEFAULT_MODE) for ds in dataset_ids}

    print(f"\n  Datasets to fit: {', '.join(dataset_ids)}")
    if len(dataset_ids) > 1:
        same = input("  Use the SAME mode for all of them? [Y/n]: ").strip().lower()
    else:
        same = 'y'

    if same in ('', 'y', 'yes'):
        mode = choose_mode()
        configure_feed(mode, cfg)
        return {ds: mode for ds in dataset_ids}

    modes = {}
    for ds in dataset_ids:
        print(f"\n  --- {ds} ---")
        modes[ds] = choose_mode()
    # Configure feed once per distinct fed-batch mode used
    for mode in sorted({m for m in modes.values() if m != 'batch'}):
        configure_feed(mode, cfg)
    return modes

# ══════════════════════════════════════════════════════════════════════════════
#  2. FEED FLOW RATE POLICY
# ══════════════════════════════════════════════════════════════════════════════
def net_volume_rate(F_S, F_N, cfg):
    """
    Net rate of broth volume change (L/h):

        F = F_S*(rho_FS - S_F)/rho_w  +  F_N*(rho_FN - N_F)/rho_w

    Each feed stream contributes only the volume of its *solvent*. The dissolved
    solute (S_F or N_F g/L) displaces water, so the effective volume added per
    litre of feed is (rho_F - C_F)/rho_w, not 1.
    """
    return (F_S * (cfg['rho_FS'] - cfg['S_F']) / cfg['rho_w']
          + F_N * (cfg['rho_FN'] - cfg['N_F']) / cfg['rho_w'])


def feed_rates(mode, t, V, X, S, N, mu_s, mu_N, cfg):
    """
    Return the feed solution flow rates (F_S, F_N) in L/h for the current state.

    Batch  ->  (0, 0), which makes F = 0 and therefore D = 0.
    """
    if mode == 'batch':
        return 0.0, 0.0

    if t < cfg['t_feed_start']:
        return 0.0, 0.0

    # ── Constant feed ─────────────────────────────────────────────────────────
    if mode == 'fedbatch_constant':
        return cfg['F_S_const'], cfg['F_N_const']

    # ── Exponential feed ──────────────────────────────────────────────────────
    if mode == 'fedbatch_exponential':
        F_tot = cfg['F0'] * np.exp(cfg['mu_set'] * (t - cfg['t_feed_start']))
        r = cfg['feed_ratio']
        F_S, F_N = r * F_tot, (1.0 - r) * F_tot
        # cap on the NET volume rate so D stays below D_max
        F_net = net_volume_rate(F_S, F_N, cfg)
        if F_net > cfg['D_max'] * V and F_net > 1e-12:
            scale = cfg['D_max'] * V / F_net
            F_S, F_N = F_S * scale, F_N * scale
        return F_S, F_N

    # ── Setpoint (substrate-stat / nutristat) feed ────────────────────────────
    # Solve for the feed rates that hold S and N at their setpoints.
    #
    #   dS/dt = 0  ->  F_S*S_F/V = mu_s*X + D*S     ->  F_S/V = (mu_s*X + D*S)/S_F
    #   dN/dt = 0  ->  F_N*N_F/V = mu_N*X + D*N     ->  F_N/V = (mu_N*X + D*N)/N_F
    #
    # Substituting into  D = F/V = [F_S*(rho_FS-S_F) + F_N*(rho_FN-N_F)] / (rho_w*V)
    # and solving for D gives:
    #
    #             X * [ mu_s*(rho_FS-S_F)*N_F + mu_N*(rho_FN-N_F)*S_F ]
    #   D = ───────────────────────────────────────────────────────────────────
    #        rho_w*S_F*N_F - (rho_FS-S_F)*S*N_F - (rho_FN-N_F)*N*S_F
    #
    if mode == 'fedbatch_setpoint':
        S_F, N_F   = cfg['S_F'], cfg['N_F']
        dS = cfg['rho_FS'] - S_F        # effective solvent density, substrate feed
        dN = cfg['rho_FN'] - N_F        # effective solvent density, nitrogen feed

        denom = cfg['rho_w'] * S_F * N_F - dS * S * N_F - dN * N * S_F
        if abs(denom) < 1e-9:
            return 0.0, 0.0

        D = X * (mu_s * dS * N_F + mu_N * dN * S_F) / denom
        D = float(np.clip(D, 0.0, cfg['D_max']))

        # Back out each stream from its own nutrient balance
        F_S = V * (mu_s * X + D * S) / S_F if S_F > 0 else 0.0
        F_N = V * (mu_N * X + D * N) / N_F if N_F > 0 else 0.0
        return max(F_S, 0.0), max(F_N, 0.0)

    raise ValueError(f"Unknown mode: {mode}")

# ══════════════════════════════════════════════════════════════════════════════
#  3. MODEL — Euler integration with feed / dilution terms
# ══════════════════════════════════════════════════════════════════════════════
def run_model(params, S0, N0, X0, P0, tf, mode='batch', cfg=FEED_CONFIG, h=0.2):
    """
    Mass balances (fed-batch general form):

        dV/dt = F                                   D   = F / V
        dX/dt = mu_x * X            - D*X      (X = RESIDUAL, PHA-free biomass)
        total biomass reported to the fit is  Xt = X + P
        dP/dt = mu_p * X            - D*P
        dS/dt = -mu_s * X + F_S*S_F/V - D*S
        dN/dt = -mu_N * X + F_N*N_F/V - D*N

    with   F = F_S*(rho_FS - S_F)/rho_w + F_N*(rho_FN - N_F)/rho_w

    Batch is recovered exactly when F_S = F_N = 0  ->  F = 0, D = 0.

    Returns t, X, P, S, N, V, D_arr
    """
    (K_S, K_IS, K_IN, K_N, K_PS, K_PIS, K_PHB, K_PIN,
     X_m, f_PHB_max, afa, beta, m_s,
     Y_PS, Y_XN, Y_XP, Y_XS,
     mu_xs_max, mu_ps_max, mu_xp_max) = params

    S_F = cfg['S_F']; N_F = cfg['N_F']
    V0  = cfg['V0'] if mode != 'batch' else 1.0

    n = int(tf / h)
    t = np.linspace(0, tf, n + 1)
    X = np.zeros(n + 1); X[0] = X0
    P = np.zeros(n + 1); P[0] = P0
    S = np.zeros(n + 1); S[0] = S0
    N = np.zeros(n + 1); N[0] = N0
    V = np.zeros(n + 1); V[0] = V0
    D_arr = np.zeros(n + 1)

    for i in range(n):
        s = max(S[i], 0.0)
        nn = max(N[i], 0.0)
        f = P[i] / X[i] if X[i] > 1e-6 else 0.0

        # ---- specific rates ----
        mu_xs = (mu_xs_max
                 * (s / (K_S + s + s**2 / K_IS))
                 * (nn / (K_N + nn + nn**2 / K_IN))
                 * max(1 - (X[i] / X_m)**afa, 0))

        mu_xp = (mu_xp_max
                 * (f / (K_PHB + f))
                 * (nn / (K_N + nn + nn**2 / K_IN))
                 * max(1 - (X[i] / X_m)**afa, 0))

        mu_ps = (mu_ps_max
                 * (s / (K_PS + s + s**2 / K_PIS))
                 * max(1 - (f / f_PHB_max)**beta, 0)
                 * (K_PIN / (nn + K_PIN)))

        mu_x = mu_xs + mu_xp                       # net biomass growth
        mu_p = mu_ps - mu_xp / Y_XP                # net PHA formation
        mu_s = mu_xs / Y_XS + mu_ps / Y_PS + m_s   # substrate uptake
        mu_N = (mu_xs + mu_xp) / Y_XN              # nitrogen uptake

        # ---- feed flow rates and dilution rate ----
        F_S, F_N = feed_rates(mode, t[i], V[i], X[i], s, nn, mu_s, mu_N, cfg)
        F = net_volume_rate(F_S, F_N, cfg)      # density-corrected volume rate
        D = F / V[i] if V[i] > 1e-9 else 0.0
        D_arr[i] = D

        # ---- Euler step ----
        V[i+1] = V[i] + h * F
        X[i+1] = max(X[i] + h * (mu_x * X[i] - D * X[i]), 0.0)
        P[i+1] = max(P[i] + h * (mu_p * X[i] - D * P[i]), 0.0)
        S[i+1] = max(S[i] + h * (-mu_s * X[i] + F_S * S_F / V[i] - D * S[i]), 0.0)
        N[i+1] = max(N[i] + h * (-mu_N * X[i] + F_N * N_F / V[i] - D * N[i]), 0.0)

    D_arr[-1] = D_arr[-2] if n > 0 else 0.0
    return t, X, P, S, N, V, D_arr

# ══════════════════════════════════════════════════════════════════════════════
#  4. BOUNDS, OBJECTIVE, OPTIMISATION
# ══════════════════════════════════════════════════════════════════════════════
bounds = [
    (0.1, 10.0), (5.0, 100.0), (1.0, 50.0), (0.01, 2.0), (0.5, 20.0), (10.0, 300.0),
    (0.01, 1.0), (0.01, 2.0), (10.0, 200.0), (1.0, 15.0), (0.5, 20.0), (0.5, 15.0),
    (0.001, 0.2), (0.05, 1.5), (2.0, 30.0), (0.1, 3.0), (0.1, 1.5),
    (0.01, 1.5), (0.05, 5.0), (0.005, 1.0),
]
param_names = ["K_S","K_IS","K_IN","K_N","K_PS","K_PIS","K_PHB","K_PIN",
               "X_m","f_PHB_max","afa","beta","m_s","Y_PS","Y_XN","Y_XP","Y_XS",
               "mu_xs_max","mu_ps_max","mu_xp_max"]

def make_objective(t_X, X_ex, t_P, P_ex, S0, N0, X0, P0, tf, mode):
    """Fit TOTAL biomass Xt = X + P against 'Biomass_Cx'.

    The ODE state X is residual (PHA-free) biomass; the gravimetric
    Biomass_Cx measurement weighs cells INCLUDING the polymer stored in
    them. Matching X alone to that number would force the residual biomass
    to account for mass that is really PHA, and would double-count it
    against the separate PHA_Cp fit.
    """
    def objective(params):
        try:
            t, X, P, *_ = run_model(params, S0, N0, X0, P0, tf, mode=mode)
            Xt = X + P                       # total biomass -> fitted to Biomass_Cx
            rX = np.sqrt(np.mean((np.interp(t_X, t, Xt) - X_ex)**2)) / (np.mean(X_ex) + 1e-9)
            rP = np.sqrt(np.mean((np.interp(t_P, t, P) - P_ex)**2)) / (np.mean(P_ex) + 1e-9)
            return rX + rP
        except Exception:
            return 1e9
    return objective

# ══════════════════════════════════════════════════════════════════════════════
#  5. SAVE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def save_dataset_result(ds_id, r):
    with open(os.path.join(OUTPUT_DIR, f'{ds_id}_params.json'), 'w') as fh:
        json.dump({'dataset': ds_id, 'organism': r['organism'], 'mode': r['mode'],
                   'R2_X': round(r['RR_X'], 6), 'R2_P': round(r['RR_P'], 6),
                   'params': {k: round(float(v), 6)
                              for k, v in zip(param_names, r['opt_params'])}},
                  fh, indent=2)
    pd.DataFrame({'time_hr': r['t_mod'], 'Xt_model': r['Xt_mod'],
                  'X_model': r['X_mod'], 'P_model': r['P_mod'],
                  'S_model': r['S_mod'], 'N_model': r['N_mod'],
                  'V_model': r['V_mod'], 'D_model': r['D_mod']}
                 ).to_csv(os.path.join(OUTPUT_DIR, f'{ds_id}_timecourse.csv'), index=False)

def update_summary(results):
    rows = []
    for ds, r in results.items():
        row = {'Dataset': ds, 'Organism': r['organism'], 'Mode': r['mode'],
               'R2_X': round(r['RR_X'], 4), 'R2_P': round(r['RR_P'], 4)}
        row.update({k: round(float(v), 5) for k, v in zip(param_names, r['opt_params'])})
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, 'summary_parameters.csv'), index=False)

# ══════════════════════════════════════════════════════════════════════════════
#  6. MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════
df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, header=HEADER_ROW)

# Resolve the fermentation mode for every dataset (prompts if interactive)
MODES_RESOLVED = resolve_modes(DATASET_IDS, FEED_CONFIG)
print("\n" + "="*66)
print("  Running with:")
for ds, m in MODES_RESOLVED.items():
    print(f"    {ds}: {m}")
print("="*66)

def extract_timecourse(sub, col):
    """Pull (time, value) pairs for one variable, dropping rows where it's missing.

    Biomass, PHA, substrate and nitrogen are not necessarily sampled on the
    same days (e.g. a PHA reading at t=15h with no matching biomass reading),
    so each variable is extracted independently against its own Time_hr
    rather than assuming one shared time array for all of them.
    """
    d = sub[['Time_hr', col]].dropna()
    return d['Time_hr'].values.astype(float), d[col].values.astype(float)


results = {}

for ds_id in DATASET_IDS:
    mode = MODES_RESOLVED[ds_id]
    sub  = df[df['Dataset_ID'] == ds_id].sort_values('Time_hr')
    t_X, X_ex = extract_timecourse(sub, 'Biomass_Cx')
    t_P, P_ex = extract_timecourse(sub, 'PHA_Cp')
    t_S, S_ex = extract_timecourse(sub, 'Substrate_Cs')
    t_N, N_ex = extract_timecourse(sub, 'Nitrogen_Cn')
    S0, N0 = float(S_ex[0]), float(N_ex[0])
    # X is residual biomass, so the measured TOTAL at t=0 splits into stored
    # PHA (P0) and the rest (X0). Seeding X0 with the full Biomass_Cx would
    # make Xt(0) = X_ex[0] + P_ex[0] and start the fit off by P0.
    P0 = float(P_ex[0])
    X0 = max(float(X_ex[0]) - P0, 1e-9)
    tf = float(max(t_X[-1], t_P[-1], t_S[-1], t_N[-1]))
    organism = sub['Organism'].iloc[0] if 'Organism' in sub.columns else ds_id

    print(f"\n{'='*62}\n{ds_id} — {organism}   [mode = {mode}]")

    res = differential_evolution(
        make_objective(t_X, X_ex, t_P, P_ex, S0, N0, X0, P0, tf, mode),
        bounds, seed=42, maxiter=400, tol=1e-5, popsize=10,
        mutation=(0.5, 1.5), recombination=0.7, workers=1, disp=False)

    opt = res.x
    t, X, P, S, N, V, Darr = run_model(opt, S0, N0, X0, P0, tf, mode=mode)
    Xt = X + P                               # total biomass (model)
    Xf, Pf = np.interp(t_X, t, Xt), np.interp(t_P, t, P)
    RR_X = 1 - np.sum((Xf-X_ex)**2)/(np.sum((X_ex-np.mean(X_ex))**2)+1e-9)
    RR_P = 1 - np.sum((Pf-P_ex)**2)/(np.sum((P_ex-np.mean(P_ex))**2)+1e-9)
    print(f"  R2_X = {RR_X:.4f}   R2_P = {RR_P:.4f}   D_max = {Darr.max():.4f} 1/h")

    results[ds_id] = dict(t_mod=t, X_mod=X, Xt_mod=Xt, P_mod=P, S_mod=S, N_mod=N,
                          V_mod=V, D_mod=Darr, t_X=t_X, X_ex=X_ex, t_P=t_P, P_ex=P_ex,
                          t_S=t_S, S_ex=S_ex, t_N=t_N, N_ex=N_ex, RR_X=RR_X, RR_P=RR_P,
                          organism=organism, mode=mode, opt_params=opt)
    save_dataset_result(ds_id, results[ds_id])
    update_summary(results)

# ══════════════════════════════════════════════════════════════════════════════
#  7. FIGURE
# ══════════════════════════════════════════════════════════════════════════════
CM = 1/2.54
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
                     'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,
                     'legend.fontsize':7.5,'axes.linewidth':0.8,'lines.linewidth':1.4})
ids = list(results)
ncol = 2 if len(ids) > 1 else 1
nrow = int(np.ceil(len(ids)/ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(19*CM, 7*CM*nrow), squeeze=False)
axes = axes.ravel()
C_X, C_P = '#1f4e9c', '#c0392b'
for k, ds in enumerate(ids):
    r, ax = results[ds], axes[k]
    ax.plot(r['t_mod'], r['Xt_mod'], '-', color=C_X, label='$X_t=X+P$ (model)')
    ax.plot(r['t_mod'], r['X_mod'], ':', color=C_X, lw=1.0, alpha=0.8, label='$X$ residual (model)')
    ax.plot(r['t_X'], r['X_ex'], 'o', mfc='none', mec=C_X, mew=1.1, ms=4.5, label='Biomass (exp.)')
    ax.plot(r['t_mod'], r['P_mod'], '-', color=C_P, label='PHA (model)')
    ax.plot(r['t_P'], r['P_ex'], 's', mfc='none', mec=C_P, mew=1.1, ms=4.5, label='PHA (exp.)')
    ax.set_xlabel('Time (h)'); ax.set_ylabel('Concentration (g L$^{-1}$)')
    ax.set_title(f"({'abcdefgh'[k]}) {ds} – {r['organism']}  [{r['mode']}]", pad=4)
    ax.text(0.97, 0.05, f"$R^2_X$ = {r['RR_X']:.4f}\n$R^2_P$ = {r['RR_P']:.4f}",
            transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='0.6', lw=0.6))
    ax.tick_params(direction='in', length=3, top=True, right=True)
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    if k == 0: ax.legend(frameon=False, loc='upper left', handlelength=1.6, labelspacing=0.3)
for j in range(len(ids), len(axes)): axes[j].axis('off')
plt.tight_layout(pad=0.5, w_pad=1.6, h_pad=1.4)
fig.savefig(os.path.join(OUTPUT_DIR, 'PHA_biomass_fit.png'), dpi=600, bbox_inches='tight')
fig.savefig(os.path.join(OUTPUT_DIR, 'PHA_biomass_fit.pdf'), bbox_inches='tight')
print(f"\nAll results saved to {OUTPUT_DIR}/")
