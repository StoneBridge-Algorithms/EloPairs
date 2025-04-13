def update_elo(R_A, R_B, r_A, r_B, K=20, epsilon=0.0005):
    # Outcome
    if r_A > r_B + epsilon:
        S_A = 1
    elif r_A < r_B - epsilon:
        S_A = 0
    else:
        S_A = 0.5
    
    # Expected outcome
    E_A = 1 / (1 + 10 ** ((R_B - R_A) / 400))
    
    # Update
    R_A_new = R_A + K * (S_A - E_A)
    R_B_new = R_B + K * ((1 - S_A) - (1 - E_A))
    
    return R_A_new, R_B_new


def compute_elo_series(df, r_col_A, r_col_B, K=20, epsilon=0.0005, init_rating=1500):
    R_A, R_B = init_rating, init_rating
    elo_A, elo_B, elo_diff = [], [], []

    for rA, rB in zip(df[r_col_A], df[r_col_B]):
        R_A, R_B = update_elo(R_A, R_B, rA, rB, K=K, epsilon=epsilon)
        elo_A.append(R_A)
        elo_B.append(R_B)
        elo_diff.append(R_A - R_B)

    df['elo_A'] = elo_A
    df['elo_B'] = elo_B
    df['elo_diff'] = elo_diff
    df['elo_momentum'] = df['elo_diff'].diff()
    df['elo_smooth'] = df['elo_diff'].ewm(span=10).mean()

    return df

