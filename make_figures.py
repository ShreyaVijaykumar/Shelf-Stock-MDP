"""Generate Figures 1-3 (before vs after MDP). Run: python make_figures.py
Needs numpy, matplotlib and inventory_mdp.py in the same folder."""
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import inventory_mdp as M

random.seed(7)
S, DAYS, N = M.STATES, M.DAYS, 3000
BEFORE, AFTER = "#c77a00", "#0f6b5c"


def imm(s, a):
    return sum(p * r for p, _, _, _, r in M.outcomes(s, a))


# "Before MDP" rules: no value function, no look-ahead
myopic = {s: max(M.actions(s), key=lambda a: imm(s, a)) for s in S}   # best reward today only
rules = {
    "Myopic\n(best reward today)": lambda s: myopic[s],
    "Random\norder": lambda s: random.choice(M.actions(s)),
}
mdp = lambda s: M.POL_STAR[s]


def week(rule, demands, s=1):
    cum, lost, left = [0.0], 0, 0
    for d in demands:
        a = rule(s)
        stock = s + a
        sold = min(d, stock)
        nxt = stock - sold
        cum.append(cum[-1] + 5 * sold - 2 * a - 0.5 * nxt - (d - sold))
        lost += d - sold
        left += nxt
        s = nxt
    return cum, lost, left


# identical demand for every manager in every simulated week
dem = [[M.sample(M.DEMAND_P) for _ in range(DAYS)] for _ in range(N)]
res = {}
for name, r in list(rules.items()) + [("With MDP\n(Bellman optimal)", mdp)]:
    runs = [week(r, d) for d in dem]
    res[name] = dict(cum=np.array([x[0] for x in runs]),
                     lost=np.array([x[1] for x in runs]),
                     left=np.array([x[2] for x in runs]))
names = list(res)
after = names[-1]
b1 = names[0]

# ---- Figure 1: cumulative profit through the week, before vs after
fig, ax = plt.subplots(figsize=(8, 5))
for n, c in [(b1, BEFORE), (after, AFTER)]:
    c_ = res[n]["cum"]
    m = c_.mean(0)
    ci = 1.96 * c_.std(0, ddof=1) / np.sqrt(N)
    ax.plot(m, color=c, lw=3, marker="o", label=n.replace("\n", " "))
    ax.fill_between(range(DAYS + 1), m - ci, m + ci, color=c, alpha=0.2)
ax.set(title=f"Figure 1. Cumulative profit: before vs after MDP (mean of {N} weeks, same demand)",
       xlabel="Day", ylabel="Cumulative profit")
ax.legend()
fig.tight_layout()
fig.savefig("images/figure1_cumulative_profit.png", dpi=150)

# ---- Figure 2: outcome metrics, before vs after
fig, ax = plt.subplots(1, 3, figsize=(12, 4.2))
cols = [BEFORE, "#b3392f", AFTER]
for i, (key, title) in enumerate([("cum", "Weekly profit"), ("lost", "Lost sales per week"),
                                  ("left", "Unit-nights left over per week")]):
    vals = [res[n][key][:, -1].mean() if key == "cum" else res[n][key].mean() for n in names]
    ax[i].bar(range(3), vals, color=cols)
    ax[i].set_xticks(range(3))
    ax[i].set_xticklabels(names, fontsize=8)
    ax[i].set_title(title)
    for j, v in enumerate(vals):
        ax[i].text(j, v, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
fig.suptitle("Figure 2. Before vs after MDP: outcome comparison")
fig.tight_layout()
fig.savefig("images/figure2_outcome_comparison.png", dpi=150)

# ---- Figure 3: decisions and values by state
Vm = M.policy_evaluation(lambda s: {myopic[s]: 1.0})
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
x = np.arange(len(S))
ax[0].bar(x - 0.2, [myopic[s] for s in S], 0.4, color=BEFORE, label="Before (myopic)")
ax[0].bar(x + 0.2, [M.POL_STAR[s] for s in S], 0.4, color=AFTER, label="After (MDP)")
ax[0].set(title="Order quantity chosen at each stock level", xlabel="Stock s", ylabel="Order a",
          xticks=x, yticks=[0, 1, 2])
ax[0].legend()
ax[1].bar(x - 0.2, [Vm[s] for s in S], 0.4, color=BEFORE, label="Before (myopic)")
ax[1].bar(x + 0.2, [M.V_STAR[s] for s in S], 0.4, color=AFTER, label="After (MDP)")
ax[1].set(title="Long-run value v(s) of each state", xlabel="Stock s", ylabel="v(s)", xticks=x, ylim=(0, 42))
ax[1].legend()
fig.suptitle("Figure 3. Decisions and values by state: before vs after MDP")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("images/figure3_decisions_and_values.png", dpi=150)

for n in names:
    print(n.replace("\n", " "), "| profit %.1f | lost %.2f | left %.2f" % (
        res[n]["cum"][:, -1].mean(), res[n]["lost"].mean(), res[n]["left"].mean()))
print("myopic policy:", myopic, "| MDP policy:", M.POL_STAR)
print("v myopic:", [round(Vm[s], 1) for s in S], "| v*:", [round(M.V_STAR[s], 1) for s in S])
