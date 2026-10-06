import numpy as np
import matplotlib
import matplotlib.pyplot as plt

SEED, N = 20260928, 200_000
rng = np.random.default_rng(SEED)
cases = {
    'Independent, exact': (np.diag([4., 1.]), np.zeros(2)),
    'Independent, noisy A': (np.diag([4., 1.]), np.array([100., 0.])),
    'Correlated, exact': (np.array([[2., 0., 0.], [0., 1.5, 1.2], [0., 1.2, 1.4]]), np.zeros(3)),
}
results = []
print(f'Seed={SEED}; samples/case={N}; NumPy={np.__version__}; Matplotlib={matplotlib.__version__}')
for name, (cov, noise) in cases.items():
    assert np.linalg.eigvalsh(cov).min() > 0
    u = np.diag(cov)
    local = u**2 / (u + noise)
    global_gain = (cov**2).sum(axis=0) / (u + noise)
    z = rng.multivariate_normal(np.zeros(len(u)), cov, size=N)
    estimates, errors = [], []
    for a in range(len(u)):
        denom = u[a] + noise[a]
        y = z[:, a] + rng.normal(0, np.sqrt(noise[a]), N)
        mean = y[:, None] * cov[:, a] / denom
        posterior = cov - np.outer(cov[:, a], cov[a, :]) / denom
        assert np.linalg.eigvalsh(posterior).min() >= -1e-12
        np.testing.assert_allclose(np.trace(cov - posterior), global_gain[a])
        np.testing.assert_allclose((cov - posterior)[a, a], local[a])
        # Direct realized error reduction, NOT a sample of the analytic gain formula.
        improvement = z**2 - (z - mean)**2
        draws = np.column_stack([improvement[:, a], improvement.sum(axis=1)])
        estimate, se = draws.mean(axis=0), draws.std(axis=0, ddof=1) / np.sqrt(N)
        assert np.all(np.abs(estimate - [local[a], global_gain[a]]) < 5 * se + 1e-12)
        estimates.append(estimate)
        errors.append(se)
    results.append((u, local, global_gain, np.array(estimates), np.array(errors)))
    print('\n' + name + ' (global values use sum loss):')
    for a, label in enumerate('ABC'[:len(u)]):
        print(f' {label}: U={u[a]:.4f}, local G={local[a]:.4f}, global G={global_gain[a]:.4f}; '
              f'MC local/global={estimates[a][0]:.4f}/{estimates[a][1]:.4f}; '
              f'SE={errors[a][0]:.4f}/{errors[a][1]:.4f}')
    print(' Winners U/local/global:', '/'.join('ABC'[np.argmax(v)] for v in [u, local, global_gain]))
assert [tuple(np.argmax(v) for v in r[:3]) for r in results] == [(0, 0, 0), (0, 1, 1), (0, 0, 1)]
print('\nPASS: analytic posterior checks, expected winners, and all 14 MC estimates within 5 SE.')

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout='constrained')
colors = ['#777777', '#e69f00', '#0072b2']
for ax, (name, _), (u, local, gain, mc, se) in zip(axes, cases.items(), results):
    x = np.arange(len(u))
    for k, (values, label) in enumerate(zip([u, local, gain], ['U: current local variance', 'G: local improvement', 'G: global improvement'])):
        positions = x + (k - 1) * .25
        ax.bar(positions, values, width=.23, color=colors[k], label=label)
        if k:
            ax.errorbar(positions, mc[:, k-1], yerr=1.96*se[:, k-1], fmt='k.', capsize=3)
    ax.set(title=name, xticks=x, xticklabels=list('ABC'[:len(u)]), xlabel='Candidate observation')
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('Variance / expected squared-error reduction (sum)')
axes[0].legend(fontsize=7, loc='upper right')
fig.suptitle('Bars: exact values. Black dots: Monte Carlo means with 95% intervals.', fontsize=11)
fig.savefig('comparison.png', dpi=180)
plt.close(fig)
