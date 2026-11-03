"""
Shelf-Stock MDP: a finite MDP for daily restocking of one perishable item.

This file is MDP DESIGN + PLANNING (dynamics are known, so value iteration is
dynamic programming, not learning). Section 6 sketches where RL would plug in.

Run:  python inventory_mdp.py          (needs only the standard library)
"""
import random

# ---------------------------------------------------------------- 1. MDP definition
CAP = 3                         # shelf capacity -> states 0..3
STATES = list(range(CAP + 1))
DEMAND_P = {0: 0.2, 1: 0.5, 2: 0.3}   # P(demand = d)
GAMMA = 0.9
DAYS = 7                        # one episode = one trading week
PRICE, ORDER_COST, HOLD_COST, STOCKOUT = 5.0, 2.0, 0.5, 1.0


def actions(s):
    """Order 0 .. (free shelf space) units."""
    return list(range(CAP - s + 1))


def outcomes(s, a):
    """All (prob, demand, sold, next_state, reward) for taking action a in s."""
    stock = s + a
    out = []
    for d, p in DEMAND_P.items():
        sold = min(d, stock)
        nxt = stock - sold
        r = PRICE * sold - ORDER_COST * a - HOLD_COST * nxt - STOCKOUT * (d - sold)
        out.append((p, d, sold, nxt, r))
    return out


# ---------------------------------------------------------------- 2. Bellman backups
def q_value(s, a, V):
    """q(s,a) = sum_d P(d) [ r + gamma * V(s') ]"""
    return sum(p * (r + GAMMA * V[nxt]) for p, _, _, nxt, r in outcomes(s, a))


def value_iteration(tol=1e-10):
    """Repeated Bellman OPTIMALITY backups -> v*, and greedy optimal policy."""
    V = {s: 0.0 for s in STATES}
    while True:
        delta = 0.0
        for s in STATES:
            new = max(q_value(s, a, V) for a in actions(s))
            delta = max(delta, abs(new - V[s]))
            V[s] = new
        if delta < tol:
            break
    policy = {s: max(actions(s), key=lambda a: q_value(s, a, V)) for s in STATES}
    return V, policy


def policy_evaluation(pi, tol=1e-10):
    """Repeated Bellman EXPECTATION backups -> v_pi.  pi(s) -> {action: prob}."""
    V = {s: 0.0 for s in STATES}
    while True:
        delta = 0.0
        for s in STATES:
            new = sum(pr * q_value(s, a, V) for a, pr in pi(s).items())
            delta = max(delta, abs(new - V[s]))
            V[s] = new
        if delta < tol:
            break
    return V


def explain_state(s, V):
    """Print the Bellman optimality equation for one state, term by term."""
    print(f"\nBellman analysis of state s={s}  (gamma={GAMMA})")
    qs = {a: q_value(s, a, V) for a in actions(s)}
    best = max(qs, key=qs.get)
    for a in actions(s):
        terms = " + ".join(f"{p:.1f}*({r:.1f} + {GAMMA}*{V[nxt]:.2f})"
                           for p, _, _, nxt, r in outcomes(s, a))
        mark = "  <-- max" if a == best else ""
        print(f"  q({s},{a}) = {terms} = {qs[a]:.3f}{mark}")
    print(f"  v*({s}) = {V[s]:.3f}, best action = order {best}")


# ---------------------------------------------------------------- 3. Policies
V_STAR, POL_STAR = value_iteration()

POLICIES = {
    "optimal":   lambda s: {POL_STAR[s]: 1.0},
    "up_to_2":   lambda s: {max(0, 2 - s): 1.0},
    "never":     lambda s: {0: 1.0},
    "random":    lambda s: {a: 1 / len(actions(s)) for a in actions(s)},
}


def sample(dist):
    u, c = random.random(), 0.0
    for k, p in dist.items():
        c += p
        if u < c:
            return k
    return k


# ---------------------------------------------------------------- 4. Simulation
def run_episode(pi, s0=1, verbose=False):
    """One trading week. Returns (discounted return G0, undiscounted profit)."""
    s, G, U = s0, 0.0, 0.0
    if verbose:
        print("day  s  a  d  sold  reward  s'")
    for t in range(DAYS):
        a = sample(pi(s))
        _, d, sold, nxt, r = (lambda o: o[sample({i: x[0] for i, x in enumerate(o)})])(outcomes(s, a))
        G += GAMMA ** t * r
        U += r
        if verbose:
            print(f"{t+1:>3} {s:>2} {a:>2} {d:>2} {sold:>5} {r:>7.1f} {nxt:>3}")
        s = nxt
    return G, U


def compare(n=2000):
    print(f"\nMonte Carlo over {n} weeks per policy (start s=1)")
    print(f"{'policy':<10}{'mean G0':>10}{'mean profit':>14}")
    for name, pi in POLICIES.items():
        res = [run_episode(pi) for _ in range(n)]
        print(f"{name:<10}{sum(g for g, _ in res)/n:>10.2f}{sum(u for _, u in res)/n:>14.2f}")


# ---------------------------------------------------------------- 5. Report
if __name__ == "__main__":
    random.seed(0)
    print("Optimal values v*(s):", {s: round(v, 3) for s, v in V_STAR.items()})
    print("Optimal policy      :", POL_STAR)
    for s in STATES:
        explain_state(s, V_STAR)

    print("\nv_pi(s) for each policy (infinite-horizon, discounted)")
    for name, pi in POLICIES.items():
        v = policy_evaluation(pi)
        print(f"  {name:<8}", [round(v[s], 2) for s in STATES])

    print("\nOne sample week under the optimal policy")
    run_episode(POLICIES["optimal"], verbose=True)
    compare()

# ---------------------------------------------------------------- 6. Future RL (not implemented here)
# Hide DEMAND_P from the agent and learn q(s,a) from interaction, e.g. tabular Q-learning:
#     Q[s][a] += alpha * (r + GAMMA * max(Q[s'].values()) - Q[s][a])
# then check that argmax_a Q[s][a] matches POL_STAR and Q approaches the q-values above.
