"""
MPC + PPC — PHA fermentation, CONTROL-AFFINE  ẋ = f(x) + g(x)·u
================================================================
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ── Parameters ──────────────────────────────────────────────────────────────────
mu_xs_max = 0.14884;  mu_xp_max = 0.49958;  mu_ps_max = 2.28684
K_S  = 8.71648;    K_IS  = 16.54088
K_N  = 1.31361;     K_IN  = 36.1489
K_PHB = 0.12512
K_PS = 19.89946;    K_PIS = 46.6806
K_PIN = 1.29648
Y_XS = 1.04905;   Y_XN = 13.34029;   Y_XP =2.8021;  Y_PS = 0.17857
m_s  = 0.02614
afa  = 0.5;    beta = 7.97204
X_m  = 30.5196;   f_PHB_max = 9.78296
#   f_PHB = P/X_residual (see kinetics), i.e. PHA per unit residual biomass.
#   That ratio is unbounded above, so f_PHB_max = 9.78 is a reachable target
#   and the self-limiting factor max(1 - (f_PHB/f_PHB_max)**beta, 0) does
#   real work: it drives mu_ps to zero exactly at f_PHB_max, making the
#   target an asymptote rather than a hard wall. beta = 7.97 is a sharp
#   exponent, so that throttle only bites in the last ~20% of the approach.
rho_FS = 1230.0; rho_FN = 1040.0; rho_w = 1000.0
S_F = 62.93;   N_F = 164.0
beta_s = (rho_FS - S_F) / rho_w     # 0.680
beta_n = (rho_FN - N_F) / rho_w     # 0.876

# ── Setpoint ────────────────────────────────────────────────────────────────────
X_REF = 2#X_m / (f_PHB_max + 1.0)  # residual-biomass target  =  2.8304 g/L
P_REF = 9.5#f_PHB_max * X_REF        # PHA target               = 27.6892 g/L

# -- Substrate setpoint ------------------------------------------------
S_REF = 5.0                      # C_S setpoint [g/L]

# ── Nitrogen setpoint ─────────────────────────────────────────────────
N_REF = 0.05                     # C_N setpoint [g/L]

# ── Volume setpoint ───────────────────────────────────────────────────
V_REF = 3.0                      # broth-volume setpoint [L]

# ── MPC settings ────────────────────────────────────────────────────────────────
h = 1.0;  N = 4;  n_sub = 5
Q_XP  =59.0     # PPC funnel weight on the PHA error (P - P_REF).
Q_XI  = 5.0     # PPC funnel weight on the substrate error (S - S_REF)
Q_XV  = 0.2     # PPC funnel weight on the volume error (V - V_REF).
Q_XN  = 0.2    # PPC funnel weight on the nitrogen error (N - N_REF).
Q_XR  = 0.2     # PPC funnel weight on the biomass error (X_res - X_REF).
Q_REF_W = 8.0   # reference arm weight.
Q_EX  = 8.0  
Q_PHA = 0.0
R_U   = np.diag([2.0, 2.0])
P_TERM = 15.0

# ── PPC funnel on the volume error  e_V = V - V_REF ────────────────────────
RHO_INF_V = 1.50      # accepted band on V late in the run [L]
L_V       = 0.02      # collapse rate

# ── PPC funnel on the nitrogen error  e_N = N - N_REF ──────────────────────

RHO_INF_N = 0.15      # accepted band on N late in the run [g/L]
L_N       = 0.03      # collapse rate

# ── PPC funnel on the biomass error  e_X = X_res - X_REF ───────────────────
RHO_INF_X = 0.5      # accepted band on X_res late in the run [g/L]
L_X       = 0.005      # collapse rate

# ── PPC funnel on the substrate error  e_S = S - S_REF ─────────────────────
RHO_INF_S = 0.30      # accepted steady-state band around S_REF
L_S       = 0.05      # collapse rate

# ── PPC funnel on the PHA error  e_P = P - P_REF ───────────────────────────
RHO_INF_P = 0.1       # accepted steady-state shortfall below P_REF [g/L]
L_P       = 0.01     # collapse rate - deliberately slow, see above
# RHO0_P is sized from the initial state, so it is set just after X0 below.
def rho_p(t):  return (RHO0_P- RHO_INF_P )* np.exp(-L_P * t) + RHO_INF_P

XI_CLIP = 8.0
def transform(e, r):
    ratio = np.clip(e / r, -1 + 1e-6, 1 - 1e-6)
    return (np.log((ratio + 1.0) / (1.0 - ratio)))

def excess(e, r):
    """Unclipped quadratic on how far outside the funnel the error is.

    transform() clips its ratio, so once |e| > r the log-barrier is a
    CONSTANT and that arm contributes no gradient at all. With one arm that
    was survivable - f_PHB never left its funnel. With two arms it is not:
    the substrate arm is far easier to satisfy (F_S injects S directly), so
    SLSQP spends the whole horizon on S, lets f_PHB fall out of its funnel,
    and is then unable to see any way back. Measured, at S_REF = 8: f_PHB
    ended at 0.120 - 1.2% of target - with S pinned at 8.00, 49/49 contained.
    Adding this term recovers f_PHB = 9.546 (97.6%) with S still 49/49.

    Zero inside the funnel, so it changes nothing while the barrier is doing
    its job and only takes over where the barrier has gone flat.
    """
    d = abs(e) - r
    return d * d if d > 0.0 else 0.0

# ── Input bounds  F_S ≥ 0, F_N ≥ 0 ──────────────────────────────────────────────
V_SCALE = 0.12   # relative to the 5.0 L reactor this schedule was tuned at (V0 = 5.0*0.12 = 0.6 L)
FS_UB = 60.0 * V_SCALE;  FN_UB = 0.5 * V_SCALE;  FN_UB_EARLY = 0.05 * V_SCALE
T_FN_EARLY = 10.0
def bounds_at(t0):
    b = []
    for i in range(N):
        t_i = t0 + i * h
        fn_cap = FN_UB if t_i >= T_FN_EARLY else FN_UB_EARLY
        b.append((0, FS_UB)); b.append((0, fn_cap))
    return b
BOUNDS = bounds_at(0.0)   # kept for reference; simulate() rebuilds this each step

# ── Simulation horizon / IC ─────────────────────────────────────────────────────
T_END = 48.0;  STEPS = int(T_END / h)
X0 = np.array([0.0, 0.13, 12.0, 0.64, 5.0 * V_SCALE])   # [P, X_res, S, N, V]

RHO0_P = (1.25 * abs(X0[0] - P_REF))      # = 34.6 g/L
RHO0_S = max(1.25 * abs(X0[2] - S_REF), 1.0)
RHO0_X = max(1.25 * abs(X0[1] - X_REF), 1.0)
RHO0_N = max(1.25 * abs(X0[3] - N_REF), 0.1)
RHO0_V = max(1.25 * abs(X0[4] - V_REF), 1.0)


# ── Kinetics & control-affine RHS ───────────────────────────────────────────────
def kinetics(x):
    P_i, X_i, S_i, N_i, V_i = x
    # E is allowed to reach 0 (no 1e-8 floor): that floor forced Xt > P
    # strictly and so capped f_PHB just below 1. With it gone the residual
    # biomass may vanish entirely, i.e. Xt = P and f_PHB = 1 exactly.
    P_i = max(P_i, 0.0);  X_i = max(X_i, 0.0)
    S_i = max(S_i, 1e-8); N_i = max(N_i, 1e-8)
    Xt_i  = X_i + P_i                      # total biomass = residual + PHA
    f=P_i / Xt_i if Xt_i > 0 else 0.0
    f_PHB = np.clip(P_i / X_i, 0.0, f_PHB_max - 1e-6) if X_i > 0 else 0.0
    inh   = max(1.0 - (min(Xt_i, X_m) / X_m) ** afa, 0.0)
    N_sat = N_i / (K_N + N_i + N_i**2 / K_IN)
    mu_xs = mu_xs_max * (S_i / (K_S + S_i + S_i**2 / K_IS)) * N_sat * inh
    mu_xp = mu_xp_max * (f_PHB / (K_PHB + f_PHB)) * N_sat * inh
    mu_ps = (mu_ps_max * (S_i / (K_PS + S_i + S_i**2 / K_PIS))
             * max(1.0 - (f_PHB / f_PHB_max) ** beta, 0.0)
             * (K_PIN / (N_i + K_PIN)))
    return mu_xs, mu_xp, mu_ps


def deriv(x, Fs, Fn):
    P_i, X_i, S_i, N_i, V_i = x
    V_i = max(V_i, 0.01);  Xt_i = X_i + P_i;      # total biomass
    mu_xs, mu_xp, mu_ps = kinetics(x)
    # Residual-biomass basis: every specific rate multiplies X_i, the
    # catalytically active biomass. PHA is made from substrate (dS/dt carries
    # -mu_ps/Y_PS), so it is NOT drawn out of the residual pool and dX/dt has
    # no -mu_ps term. The mu_xp pair is the one genuine handshake: dP/dt loses
    # mu_xp/Y_XP of polymer, dX/dt gains mu_xp of biomass, so Y_XP is a proper
    # biomass-per-PHA yield.
    fx = np.array([
        (mu_ps - mu_xp / Y_XP) * X_i,
        (mu_xs + mu_xp) * X_i,
        -(mu_xs / Y_XS + mu_ps / Y_PS + m_s) * X_i,
        -((mu_xs + mu_xp) / Y_XN) * X_i,
        0.0,
    ])
    gx = np.array([
        [-beta_s * P_i / V_i,        -beta_n * P_i / V_i],
        [-beta_s * X_i / V_i,        -beta_n * X_i / V_i],
        [(S_F - beta_s * S_i) / V_i, -beta_n * S_i / V_i],
        [-beta_s * N_i / V_i,        (N_F - beta_n * N_i) / V_i],
        [beta_s,                     beta_n],
    ])
    return fx + gx @ np.array([Fs, Fn])


def rk4_step(x, Fs, Fn, dt, nsub):
    ts = dt / nsub
    for _ in range(nsub):
        k1 = deriv(x,           Fs, Fn)
        k2 = deriv(x + ts/2*k1, Fs, Fn)
        k3 = deriv(x + ts/2*k2, Fs, Fn)
        k4 = deriv(x + ts*k3,   Fs, Fn)
        x  = x + ts/6 * (k1 + 2*k2 + 2*k3 + k4)
        # E floor is 0.0, not 1e-8: the old floor kept Xt strictly above P
        # and so held f_PHB = P/Xt strictly below 1.
        x  = np.maximum(x, [0.0, 0.0, 0.0, 0.0, 0.01])
    return x


# ── MPC cost ────────────────────────────────────────────────────────────────────
def mpc_cost(u_flat, x0, t0):
    u_seq = u_flat.reshape(N, 2)
    x = x0.copy();  J = 0.0
    for i in range(N):
        P_i, X_i, S_i, N_i, V_i = x
        t_i = t0 + i * h;  u = u_seq[i]
        J += u @ R_U @ u
        # PPC arm 1 — drive P onto the constant P_REF.
        J += Q_XP *transform(P_i - P_REF, rho_p(t_i))**2
      
        # PPC arm 2 — hold substrate at S_REF.
        J += Q_XI *transform(S_i - S_REF, rho_s(t_i))**2
        
        # PPC arm 3 — residual biomass toward X_REF, lightly weighted. The
        J += Q_XR *transform(X_i - X_REF, rho_x(t_i))**2
        
        # PPC arm 4 — nitrogen, also lightly weighted. This is the arm that
        J += Q_XN *transform(N_i - N_REF, rho_n(t_i))**2
       
        # PPC arm 5 — broth volume, lightly weighted. V is monotone
        J += Q_XV *transform(V_i - V_REF, rho_v(t_i))**2
        
        # optional PHA-rate reward (Q_PHA = 0 by default)
        _, _, mu_ps = kinetics(x)
        J -= Q_PHA * mu_ps * X_i
        x = rk4_step(x, u[0], u[1], h, n_sub)
    # terminal
    t_N = t0 + N * h
    
    for q_w, e_w, r_w in ((Q_XP, x[0] - P_REF, rho_p(t_N)),
                          (Q_XI, x[2] - S_REF, rho_s(t_N)),
                          (Q_XR, x[1] - X_REF, rho_x(t_N)),
                          (Q_XN, x[3] - N_REF, rho_n(t_N)),
                          (Q_XV, x[4] - V_REF, rho_v(t_N))):
        J += (P_TERM / Q_REF_W) * (q_w *transform(e_w, r_w)**2)
                                   #+ q_w / Q_REF_W * Q_EX * excess(e_w, r_w))
    return J


# ── Simulation loop ─────────────────────────────────────────────────────────────
def simulate():
    times = np.arange(STEPS + 1) * h
    xs = np.zeros((STEPS + 1, 5));  us = np.zeros((STEPS, 2))
    mu_ps_log = np.zeros(STEPS)
    xs[0] = X0;  u_prev = np.tile([0.0, 0.0], N)

    hdr = (f"{'t':>4}  {'P':>6}  {'Xres':>6}  {'S':>6}  {'N':>6}  "
           f"{'Xt':>6}  {'f_PHB':>7}  {'FS':>6}  {'FN':>6}")
    print(hdr);  print("─" * len(hdr))

    for k in range(STEPS):
        res = minimize(mpc_cost, u_prev, args=(xs[k], k * h),
                       method='SLSQP', bounds=bounds_at(k * h),
                       options={'ftol': 1e-9, 'maxiter': 500})
        u_opt = res.x.reshape(N, 2)
        Fs_k, Fn_k = u_opt[0];  us[k] = [Fs_k, Fn_k]
        u_prev = np.roll(u_opt, -1, axis=0).flatten();  u_prev[-2:] = u_opt[-1]
        _, _, mu_ps = kinetics(xs[k]);  mu_ps_log[k] = mu_ps
        xs[k+1] = rk4_step(xs[k], Fs_k, Fn_k, h, n_sub)
        if k % 3 == 0:
            P_i, X_i, S_i, N_i, V_i = xs[k]
            f_k = P_i / X_i if X_i > 1e-12 else 0.0
            print(f"{k*h:4.0f}  {P_i:6.3f}  {X_i:6.3f}  {S_i:6.3f}  {N_i:6.3f}  "
                  f"{X_i+P_i:6.3f}  {f_k:7.3f}  {Fs_k:6.4f}  {Fn_k:6.4f}")

    P_i, X_i, S_i, N_i, V_i = xs[-1]
    f_traj = np.array([xs[k,0] / xs[k,1] if xs[k,1] > 1e-12 else 0.0
                       for k in range(STEPS+1)])
    p_in = sum(abs(xs[k,0] - P_REF) <= rho_p(k*h) + 1e-9
               for k in range(STEPS+1))
    print(f"\nFinal:  X_res = {X_i:.3f} g/L   C_S = {S_i:.3f} g/L   "
          f"C_N = {N_i:.4f} g/L   P (PHA) = {P_i:.2f} g/L   "
          f"Xt (total) = {X_i + P_i:.2f} g/L")
    print(f"P     = {P_i:.3f} g/L  (target P_REF = {P_REF:.3f}, "
          f"e_P = {P_i - P_REF:+.3f}, {100.0 * P_i / P_REF:.1f}% of target); "
          f"inside funnel : {p_in}/{STEPS+1} steps")
    x_in = int(sum(abs(xs[k,1] - X_REF) <= rho_x(k*h) + 1e-9
                   for k in range(STEPS+1)))
    print(f"X_res = {X_i:.3f} g/L  (target X_REF = {X_REF:.3f}, "
          f"e_X = {X_i - X_REF:+.3f}, {100.0 * X_i / X_REF:.1f}% of target); "
          f"inside funnel : {x_in}/{STEPS+1} steps")
    print(f"Xt    = {X_i + P_i:.3f} g/L  (ceiling X_m = {X_m:.3f}, "
          f"{100.0 * (X_i + P_i) / X_m:.1f}% of it)")
    print(f"f_PHB = P/X_res = {f_traj[-1]:.3f}   "
          f"(kinetic ceiling f_PHB_max = {f_PHB_max:.3f}, "
          f"{100.0 * f_traj[-1] / f_PHB_max:.1f}% of it)")
    s_in = int(sum(abs(xs[k,2] - S_REF) <= rho_s(k*h) + 1e-9
                   for k in range(STEPS+1)))
    print(f"C_S   = {xs[-1,2]:.3f} g/L  (target S_REF = {S_REF:.2f}, "
          f"e_S = {xs[-1,2] - S_REF:+.3f}); "
          f"inside funnel : {s_in}/{STEPS+1} steps")
    n_in = int(sum(abs(xs[k,3] - N_REF) <= rho_n(k*h) + 1e-9
                   for k in range(STEPS+1)))
    print(f"C_N   = {xs[-1,3]:.4f} g/L (target N_REF = {N_REF:.3f}, "
          f"e_N = {xs[-1,3] - N_REF:+.4f}); "
          f"inside funnel : {n_in}/{STEPS+1} steps")
    v_in = int(sum(abs(xs[k,4] - V_REF) <= rho_v(k*h) + 1e-9
                   for k in range(STEPS+1)))
    print(f"V     = {xs[-1,4]:.3f} L    (target V_REF = {V_REF:.2f}, "
          f"e_V = {xs[-1,4] - V_REF:+.3f}); "
          f"inside funnel : {v_in}/{STEPS+1} steps")
    print(f"PHA mass fraction P/Xt = {P_i/(X_i+P_i):.3f}")
    return times, xs, us, mu_ps_log


# ── Plot ────────────────────────────────────────────────────────────────────────
def plot(times, xs, us, mu_ps_log, out_png):
    BLUE='#2166ac'; PUR='#762a83'; GRN='forestgreen'
    t_u = times[:-1]
    fig, axes = plt.subplots(4, 1, figsize=(10, 15), sharex=True)
    fig.suptitle("MPC + PPC on the PHA content   "
                 r"($\dot{x}=f(x)+g(x)u$,  $x=[P,X_{res},S,N,V]$)"
                 "\n"
                 rf"Five PPC funnels:  $P\to{P_REF:.2f}$,  "
                 rf"$C_S\to{S_REF:g}$,  $X_{{res}}\to{X_REF:.2f}$,  "
                 rf"$C_N\to{N_REF:g}$ g/L,  $V\to{V_REF:g}$ L",
                 fontsize=10)

    # 1 — f_PHB inside its PPC funnel (the only tracked variable)
    ax = axes[0]
    rp_ = np.array([rho_p(t) for t in times])
    ax.fill_between(times, np.maximum(P_REF - rp_, 0.0), P_REF + rp_,
                    alpha=0.15, color=GRN, label=r'funnel $P_{ref}\pm\rho_P(t)$')
    ax.plot(times, P_REF + rp_, '--', color=GRN, lw=1.2)
    ax.plot(times, np.maximum(P_REF - rp_, 0.0), '--', color=GRN, lw=1.2)
    ax.axhline(P_REF, ls=':', color='grey', lw=1.8,
               label=rf'$P_{{ref}}={P_REF:.2f}$ g/L')
    ax.plot(times, xs[:,0], color=PUR, lw=2.6, label='$P$ [g/L]')
    ax.set_ylabel('$P$ [g/L]'); ax.grid(alpha=0.3)
    ax.set_ylim(0, (P_REF + rp_[0]) * 1.05)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_title('PPC arm — PHA driven to $P_{ref}$', fontsize=10)

    # 2 — substrate inside its PPC funnel; biomass free, for context
    ax = axes[1]
    rs_ = np.array([rho_s(t) for t in times])
    ax.fill_between(times, np.maximum(S_REF - rs_, 0.0), S_REF + rs_,
                    alpha=0.18, color=BLUE, label=r'funnel $\pm\rho_S$')
    ax.plot(times, S_REF + rs_, '--', color=BLUE, lw=1.0)
    ax.plot(times, np.maximum(S_REF - rs_, 0.0), '--', color=BLUE, lw=1.0)
    ax.axhline(S_REF, color=BLUE, ls=':', lw=1.2,
               label=rf'$S_{{ref}}={S_REF:g}$')
    ax.plot(times, xs[:,2], color='crimson', lw=2.2, label='$C_S$ [g/L]')
    ax.set_ylabel('$C_S$ [g/L]'); ax.grid(alpha=0.3); ax.set_ylim(bottom=0)
    ax2 = ax.twinx()
    rx_ = np.array([rho_x(t) for t in times])
    ax2.fill_between(times, np.maximum(X_REF - rx_, 0.0), X_REF + rx_,
                     alpha=0.10, color=GRN, label=r'funnel $\pm\rho_X$ (loose)')
    ax2.plot(times, X_REF + rx_, '--', color=GRN, lw=0.9, alpha=0.6)
    ax2.plot(times, np.maximum(X_REF - rx_, 0.0), '--', color=GRN, lw=0.9,
             alpha=0.6)
    ax2.axhline(X_REF, color=GRN, ls=':', lw=1.2,
                label=rf'$X_{{ref}}={X_REF:.2f}$')
    ax2.plot(times, xs[:,1], color=GRN, lw=2.0, label='$X_{res}$ [g/L]')
    ax2.set_ylabel('$X_{res}$ [g/L]', color=GRN)
    ax2.tick_params(axis='y', labelcolor=GRN); ax2.set_ylim(bottom=0)
    l1,b1 = ax.get_legend_handles_labels(); l2,b2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, b1+b2, fontsize=8, loc='upper right')
    ax.set_title('PPC arms — substrate at $S_{ref}$ (left); '
                 'biomass in its loose funnel (right)', fontsize=10)

    # 3 — P, X, V
    ax = axes[2]; Xv = xs[:,0]+xs[:,1]
    ax.plot(times, Xv, color=GRN, lw=2, label='$X_t=X_{res}+P$ [g/L]')
    ax.plot(times, xs[:,0], color=PUR, lw=2.5, label='$P$ PHA [g/L]')
    ax.fill_between(times, xs[:,0], Xv, alpha=0.10, color=GRN)
    ax2 = ax.twinx()
    rv_ = np.array([rho_v(t) for t in times])
    ax2.fill_between(times, np.maximum(V_REF - rv_, 0.0), V_REF + rv_,
                     alpha=0.10, color='navy', label=r'funnel $\pm\rho_V$ (loose)')
    ax2.plot(times, V_REF + rv_, '--', color='navy', lw=0.9, alpha=0.6)
    ax2.plot(times, np.maximum(V_REF - rv_, 0.0), '--', color='navy', lw=0.9,
             alpha=0.6)
    ax2.axhline(V_REF, color='navy', ls=':', lw=1.2,
                label=rf'$V_{{ref}}={V_REF:g}$ L')
    ax2.plot(times, xs[:,4], color='navy', lw=1.6, ls='-.', alpha=0.9, label='$V$ [L]')
    ax2.set_ylabel('$V$ [L]', color='navy'); ax2.tick_params(axis='y', labelcolor='navy')
    # f = P/Xt on its own outward-offset spine. This is the PHA MASS FRACTION
    # (0-1), not the controlled variable f_PHB = P/X_res (0-9.78) shown in
    # panel 1 - same polymer, different denominator, so it must not share an
    # axis with either the g/L traces or V.
    ax3 = ax.twinx(); ax3.spines['right'].set_position(('outward', 52))
    fv3 = np.divide(xs[:,0], Xv, out=np.zeros_like(Xv), where=Xv > 1e-12)
    ax3.plot(times, fv3, color='crimson', lw=1.8, ls='--',
             label='$f=P/X_t$ [g/g]')
    ax3.set_ylabel('$f=P/X_t$ [g/g]', color='crimson')
    ax3.tick_params(axis='y', labelcolor='crimson')
    ax3.set_ylim(0, 1.05)
    l1,b1 = ax.get_legend_handles_labels(); l2,b2 = ax2.get_legend_handles_labels()
    l3,b3 = ax3.get_legend_handles_labels()
    ax.legend(l1+l2+l3, b1+b2+b3, fontsize=9, loc='upper left')
    ax.set_ylabel('[g/L]'); ax.grid(alpha=0.3)
    ax.set_title('Total biomass $X_t$, PHA $P$ (shaded = residual $X_{res}$), '
                 'volume and PHA mass fraction', fontsize=10)

    # 4 — feeds + nitrogen
    ax = axes[3]
    ax.step(t_u, us[:,0], where='post', color='seagreen', lw=2, label='$F_S$ [L/h]')
    ax.step(t_u, us[:,1], where='post', color=BLUE, lw=2, label='$F_N$ [L/h]')
    ax2 = ax.twinx()
    rn_ = np.array([rho_n(t) for t in times])
    ax2.fill_between(times, np.maximum(N_REF - rn_, 0.0), N_REF + rn_,
                     alpha=0.12, color='darkorange',
                     label=r'funnel $\pm\rho_N$ (loose)')
    ax2.plot(times, N_REF + rn_, '--', color='darkorange', lw=0.9, alpha=0.6)
    ax2.plot(times, np.maximum(N_REF - rn_, 0.0), '--', color='darkorange',
             lw=0.9, alpha=0.6)
    ax2.axhline(N_REF, color='darkorange', ls=':', lw=1.2,
                label=rf'$N_{{ref}}={N_REF:g}$')
    ax2.plot(times, xs[:,3], color='darkorange', lw=1.8, label='$N$ [g/L]')
    ax2.set_ylabel('$N$ [g/L]', color='darkorange'); ax2.tick_params(axis='y', labelcolor='darkorange')
    l1,b1 = ax.get_legend_handles_labels(); l2,b2 = ax2.get_legend_handles_labels()
    ax.legend(l1+l2, b1+b2, fontsize=9, loc='upper right')
    ax.set_ylabel('feed [L/h]'); ax.set_xlabel('time [h]'); ax.grid(alpha=0.3)
    ax.set_title(r'Feeds and $N$ — $F_N\approx0$ early early, then trickled: nitrogen only ever works against $f_{PHB}$', fontsize=9.5)

    plt.tight_layout()
    plt.savefig(out_png, dpi=130, bbox_inches='tight')
    print(f"\nPlot saved → {out_png}")


def export_excel(times, xs, us, mu_ps_log, out_xlsx):

    P, Xr, S, Nv, V = (xs[:, i] for i in range(5))   # Nv: N is the MPC horizon
    Xt  = Xr + P
    f   = np.divide(P, Xr, out=np.zeros_like(P), where=Xr > 1e-12)
    rho   = np.array([rho_p(t) for t in times])
    rho_S = np.array([rho_s(t) for t in times])
    rho_X = np.array([rho_x(t) for t in times])
    rho_N = np.array([rho_n(t) for t in times])
    rho_V = np.array([rho_v(t) for t in times])
    e_P   = P - P_REF
    pad = np.full(len(times) - len(us), np.nan)

    traj = pd.DataFrame({
        'time_h':            times,
        'f_PHB_P_over_Xres': f,
        'P_over_Xt':         np.divide(P, Xt, out=np.zeros_like(P), where=Xt > 1e-12),
        'P_PHA_g_L':         P,
        'X_res_g_L':         Xr,
        'Xt_total_g_L':      Xt,
        'S_g_L':             S,
        'N_g_L':             Nv,
        'V_L':               V,
        'PHA_mass_g':        P * V,
        'e_P_g_L':           e_P,
        'rho_p':             rho,
        'P_funnel_lo':       P_REF - rho,
        'P_funnel_hi':       P_REF + rho,
        'rho_s':             rho_S,
        'S_funnel_lo':       S_REF - rho_S,
        'S_funnel_hi':       S_REF + rho_S,
        'rho_x':             rho_X,
        'X_funnel_lo':       X_REF - rho_X,
        'X_funnel_hi':       X_REF + rho_X,
        'rho_n':             rho_N,
        'N_funnel_lo':       N_REF - rho_N,
        'N_funnel_hi':       N_REF + rho_N,
        'rho_v':             rho_V,
        'V_funnel_lo':       V_REF - rho_V,
        'V_funnel_hi':       V_REF + rho_V,
        'F_S_L_h':           np.concatenate([us[:, 0], pad]),
        'F_N_L_h':           np.concatenate([us[:, 1], pad]),
        'mu_ps':             np.concatenate([mu_ps_log, pad]),
    })

    p_in = int(np.sum(np.abs(e_P) <= rho + 1e-9))
    s_in = int(np.sum(np.abs(S - S_REF) <= rho_S + 1e-9))
    x_in = int(np.sum(np.abs(Xr - X_REF) <= rho_X + 1e-9))
    n_in = int(np.sum(np.abs(Nv - N_REF) <= rho_N + 1e-9))
    v_in = int(np.sum(np.abs(V - V_REF) <= rho_V + 1e-9))
    summary = pd.DataFrame([{
        'P_REF':             P_REF,
        'P_final_g_L':       P[-1],
        'e_P_final_g_L':     e_P[-1],
        'P_percent_of_ref':  100.0 * P[-1] / P_REF,
        'P_in_funnel_steps': "%d/%d" % (p_in, len(times)),
        'S_REF':             S_REF,
        'S_in_funnel_steps': "%d/%d" % (s_in, len(times)),
        'X_REF':             X_REF,
        'X_in_funnel_steps': "%d/%d" % (x_in, len(times)),
        'N_REF':             N_REF,
        'N_in_funnel_steps': "%d/%d" % (n_in, len(times)),
        'V_REF':             V_REF,
        'V_in_funnel_steps': "%d/%d" % (v_in, len(times)),
        'X_res_final':       Xr[-1],
        'Xt_percent_of_X_m': 100.0 * Xt[-1] / X_m,
        'f_PHB_final':       f[-1],
        'f_PHB_percent_max': 100.0 * f[-1] / f_PHB_max,
        'P_final_g_L':       P[-1],
        'X_res_final_g_L':   Xr[-1],
        'Xt_final_g_L':      Xt[-1],
        'P_over_Xt_final':   P[-1] / Xt[-1] if Xt[-1] > 0 else 0.0,
        'S_final_g_L':       S[-1],
        'N_final_g_L':       Nv[-1],
        'V_final_L':         V[-1],
        'total_PHA_mass_g':  P[-1] * V[-1],
        'F_S_total_L':       us[:, 0].sum() * h,
        'F_N_total_L':       us[:, 1].sum() * h,
        'RU_FS':             float(R_U[0, 0]),
        'RU_FN':             float(R_U[1, 1]),
        'Q_XP':              Q_XP,
        'Q_XI':              Q_XI,
        'Q_XR':              Q_XR,
        'Q_XN':              Q_XN,
        'Q_XV':              Q_XV,
        'Q_EX':              Q_EX,
        'P_TERM':            P_TERM,
        'horizon_N':         N,
        'RHO0_P':            RHO0_P,
        'RHO_INF_P':         RHO_INF_P,
        'L_P':               L_P,
        'RHO0_S':            RHO0_S,
        'RHO_INF_S':         RHO_INF_S,
        'L_S':               L_S,
        'RHO0_X':            RHO0_X,
        'RHO_INF_X':         RHO_INF_X,
        'L_X':               L_X,
        'RHO0_N':            RHO0_N,
        'RHO_INF_N':         RHO_INF_N,
        'L_N':               L_N,
        'RHO0_V':            RHO0_V,
        'RHO_INF_V':         RHO_INF_V,
        'L_V':               L_V,
        'FS_UB':             FS_UB,
        'FN_UB':             FN_UB,
        'T_END_h':           T_END,
    }])

    with pd.ExcelWriter(out_xlsx, engine='openpyxl') as xl:
        traj.to_excel(xl, sheet_name='trajectory', index=False)
        summary.to_excel(xl, sheet_name='summary', index=False)
    print("Excel saved -> " + str(out_xlsx))


if __name__ == '__main__':
    res = simulate()
    HERE = os.path.dirname(os.path.abspath(__file__))
    stem = os.path.splitext(os.path.basename(__file__))[0]
    plot(*res, out_png=os.path.join(HERE, stem + '.png'))
    export_excel(*res, out_xlsx=os.path.join(HERE, stem + '.xlsx'))
