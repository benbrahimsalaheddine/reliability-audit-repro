# Common constants for every computation in the paper (S-1, 12 Nov 2026). Author's order: same seed, N and B everywhere.
SEED = 0        # every result starts its own numpy default_rng(SEED)
N = 5000        # simulation replications (author, 12 Nov 2026: N = 5,000 for rate simulations; 95% interval of a 5% rate is about +-0.6 point)
B = 10000       # bootstrap draws
