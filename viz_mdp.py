"""Generate Figure 4 (Bellman analysis). Run: python viz_mdp.py  (needs matplotlib, numpy)
Imports the MDP from inventory_mdp.py (keep both files in the same folder)."""
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import inventory_mdp as M

random.seed(1)
S, A = M.STATES, range(M.CAP + 1)

# A. Bellman optimality backups: v_k(s) per sweep, starting from zeros
V = {s: 0.0 for s in S}
hist = [[V[s] for s in S]]
for _ in range(60):
    V = {s: max(M.q_value(s, a, V) for a in M.actions(s)) for s in S}
    hist.append([V[s] for s in S])
hist = np.array(hist)

# B. converged q*(s,a)
Q = np.full((len(S), len(A)), np.nan)
for s in S:
    for a in M.actions(s):
        Q[s, a] = M.q_value(s, a, M.V_STAR)

# C. Monte Carlo returns per policy
N = 3000
stats = {}
for name, pi in M.POLICIES.items():
    g = np.array([M.run_episode(pi)[0] for _ in range(N)])
    stats[name] = (g.mean(), 1.96 * g.std(ddof=1) / np.sqrt(N))

# D. myopic vs far-sighted at s = 0
s0 = 0
imm = [sum(p * r for p, _, _, _, r in M.outcomes(s0, a)) for a in M.actions(s0)]
qs = [M.q_value(s0, a, M.V_STAR) for a in M.actions(s0)]

fig, ax = plt.subplots(2, 2, figsize=(12, 8.5))
fig.suptitle("Figure 4. Bellman analysis of the Shelf-Stock MDP", fontsize=14)

for s in S:
    ax[0, 0].plot(hist[:, s], label=f"v(s={s})")
ax[0, 0].set(title="A. Bellman backups converge to v*", xlabel="sweep k", ylabel="value")
ax[0, 0].legend()

im = ax[0, 1].imshow(Q, cmap="viridis", origin="lower")
for s in S:
    best = int(np.nanargmax(Q[s]))
    for a in A:
        if not np.isnan(Q[s, a]):
            ax[0, 1].text(a, s, f"{Q[s, a]:.1f}" + ("\n(best)" if a == best else ""),
                          ha="center", va="center", color="w", fontsize=9,
                          fontweight="bold" if a == best else "normal")
ax[0, 1].set(title="B. q*(s,a): best action per stock level", xlabel="order a", ylabel="stock s",
             xticks=list(A), yticks=S)
fig.colorbar(im, ax=ax[0, 1], shrink=0.8)

names = list(stats)
ax[1, 0].bar(names, [stats[n][0] for n in names], yerr=[stats[n][1] for n in names], capsize=5,
             color=["#0f6b5c", "#4fa38f", "#c77a00", "#b3392f"])
ax[1, 0].set(title=f"C. Mean discounted return, {N} simulated weeks (95% CI)", ylabel="G0")
ax[1, 0].tick_params(axis="x", rotation=15)

x = np.arange(len(imm))
imm_gap = np.array(imm) - max(imm)      # 0 = the action this criterion would pick
q_gap = np.array(qs) - max(qs)
ax[1, 1].bar(x - 0.2, imm_gap, 0.4, label="myopic: immediate reward", color="#c77a00")
ax[1, 1].bar(x + 0.2, q_gap, 0.4, label="far-sighted: q*(0,a)", color="#0f6b5c")
ax[1, 1].set(title="D. Empty shelf: shortfall vs each rule's pick (0 = pick)",
             xlabel="order a", xticks=x, ylabel="value minus best")
ax[1, 1].legend(fontsize=8, loc="center")

fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig("images/figure4_bellman_analysis.png", dpi=150)
print("myopic best a =", int(np.argmax(imm)), "| q* best a =", int(np.argmax(qs)))
for n, (m, ci) in stats.items():
    print(f"{n:<8} {m:6.2f} +/- {ci:.2f}")
