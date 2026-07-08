class 2DRNDA:

    def __init__(
        self, u_dim, v_dim, mu=0.1, gamma1=0.1, gamma2=0.1, theta=0.1, delta=0.1
    ):
        self.u_dim = u_dim
        self.v_dim = v_dim
        self.mu = mu
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.theta = theta
        self.delta = delta
        self.L = None
        self.R = None

    def fit(self, X_train, y_train, rnn_data, max_out=5, max_in=3):
        n, h, w = X_train.shape
        self.L = np.eye(h, self.u_dim)
        self.R = np.eye(w, self.v_dim)

        for out_it in range(max_out):
            self.L = self._update_projection(
                X_train,
                y_train,
                rnn_data,
                fixed_mat=self.R,
                current_P=self.L,
                gamma=self.gamma1,
                mode="left",
                max_in=max_in,
            )

            self.R = self._update_projection(
                X_train,
                y_train,
                rnn_data,
                fixed_mat=self.L,
                current_P=self.R,
                gamma=self.gamma2,
                mode="right",
                max_in=max_in,
            )
        return self

    def _update_projection(
        self, X, y, rnn_data, fixed_mat, current_P, gamma, mode, max_in
    ):
        sw_list, sb_list = self._compute_2d_scatters(
            X, y, rnn_data, fixed_mat, mode
        )
        m_views = len(sw_list)
        dim = sw_list[0].shape[0]
        target_dim = self.u_dim if mode == "left" else self.v_dim

        P = current_P
        alpha = np.ones(m_views) / m_views
        beta = np.ones(m_views) / m_views

        for in_it in range(max_in):
            M_base = np.zeros((dim, dim))
            for i in range(m_views):
                M_base += alpha[i] * sw_list[i] - self.mu * beta[i] * sb_list[i]

            row_norms = np.linalg.norm(P, axis=1)
            D = np.diag(1.0 / (2.0 * row_norms + 1e-10))

            eig_vals, eig_vecs = la.eigh(M_base + gamma * D)
            P = eig_vecs[:, :target_dim]

            w_vals = np.array([np.trace(P.T @ Sw @ P) for Sw in sw_list])
            alpha = self._update_weights(w_vals, self.theta, mode_type="min")

            b_vals = np.array([np.trace(P.T @ Sb @ P) for Sb in sb_list])
            beta = self._update_weights(b_vals, self.delta, mode_type="max")
        return P

    def _compute_2d_scatters(self, X, y, rnn_data, fixed_mat, mode):
        sw_l, sb_l = [], []
        for rnn_map, valid_idx in rnn_data:
            n, h, w = X.shape
            dim = h if mode == "left" else w
            Sw = np.zeros((dim, dim))
            Sb = np.zeros((dim, dim))

            m_tildes = {i: np.mean(X[rnn_map[i]], axis=0) for i in valid_idx}

            for i in valid_idx:
                for q in rnn_map[i]:
                    diff = X[q] - m_tildes[i]
                    if mode == "left":
                        Sw += diff @ fixed_mat @ fixed_mat.T @ diff.T
                    else:
                        Sw += diff.T @ fixed_mat @ fixed_mat.T @ diff

                for j in valid_idx:
                    if y[i] != y[j]:
                        diff_b = m_tildes[i] - m_tildes[j]
                        if mode == "left":
                            Sb += diff_b @ fixed_mat @ fixed_mat.T @ diff_b.T
                        else:
                            Sb += diff_b.T @ fixed_mat @ fixed_mat.T @ diff_b

            if np.trace(Sw) > 0:
                Sw /= np.trace(Sw)
            if np.trace(Sb) > 0:
                Sb /= np.trace(Sb)
            sw_l.append(Sw)
            sb_l.append(Sb)
        return sw_l, sb_l

    def _update_weights(self, vals, param, mode_type):
        m = len(vals)
        if mode_type == "min":
            raw = 1.0 / m + (np.mean(vals) - vals) / (2 * param + 1e-10)
        else:
            raw = 1.0 / m + (vals - np.mean(vals)) / (2 * param + 1e-10)
        w = np.maximum(raw, 0)
        return w / (np.sum(w) + 1e-10)

    def transform(self, X):
        return np.array([(self.L.T @ img @ self.R).flatten() for img in X])



def prepare_rnn(X, y, k_list, t=2):
    n = X.shape[0]
    X_f = X.reshape(n, -1)
    res = []
    for k in k_list:
        rnn_map = {i: [] for i in range(n)}
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            if len(idx) <= k:
                continue
            nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X_f[idx])
            idxs = nbrs.kneighbors(X_f[idx], return_distance=False)
            for lq, neighbors in enumerate(idxs):
                q = idx[lq]
                for lp in neighbors[1:]:
                    rnn_map[idx[lp]].append(q)
        valid = [i for i in range(n) if len(rnn_map[i]) >= t]
        res.append((rnn_map, valid))
    return res