"""Honest forest implementation speed versus scikit-learn forest"""
# Authors: Ronan Perry
# Adopted from: https://github.com/rflperry/ProgLearn/blob/UF/
# License: MIT

import numpy as np
import timeit
import matplotlib.pyplot as plt
from honest_forests import HonestForestClassifier
from sklearn.ensemble import RandomForestClassifier


n_estimators = 100
n_jobs = 1

hf = HonestForestClassifier(
        n_estimators=n_estimators,
        n_jobs=n_jobs,
    )

rf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs)

d = 50
n_timeits = 10
ns = [100, 1000, 10000]

results = []
for n in ns:
    n_list = []
    X = np.random.normal(0, 1, (n, d))
    y = np.random.binomial(1, 0.5, n)

    rf = rf.fit(X, y)
    hf = hf.fit(X, y)

    # RF fit, predict
    time = timeit.timeit(lambda: rf.fit(X, y), number=n_timeits)
    n_list.append(time / n_timeits)
    time = timeit.timeit(lambda: rf.predict_proba(X), number=n_timeits)
    n_list.append(time / n_timeits)

    # Honest fit, predict
    time = timeit.timeit(lambda: hf.fit(X, y), number=n_timeits)
    n_list.append(time / n_timeits)
    time = timeit.timeit(lambda: hf.predict_proba(X), number=n_timeits)
    n_list.append(time / n_timeits)

    results.append(n_list)


columns = [
    'RandomForest',
    'HonestForest',
]
means = np.asarray(results)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), facecolor='white', sharey=True, sharex=True)

for col, label in enumerate(columns):
    ax1.plot(ns, means[:, 2*col], label=label)

ax1.legend()
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_title('Fit walltimes (100 trees)')
ax1.set_ylabel('Seconds')
ax1.set_xlabel('Sample size')

for col, label in enumerate(columns):
    ax2.plot(ns, means[:, col+1], label=label)

ax2.set_xscale('log')
ax2.set_xscale('log')
ax2.set_title('Predict walltimes (100 trees)')
ax2.set_xlabel('Sample size')

plt.tight_layout()
plt.savefig("./figures/profile_speeds.png")
