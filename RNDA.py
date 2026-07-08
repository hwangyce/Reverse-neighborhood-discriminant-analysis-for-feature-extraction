import os
import numpy as np
import scipy.linalg as la
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from PIL import Image
import time
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ==========================================
# 1. RNDA
# ==========================================
class RNDA:
    def __init__(self, n_components, mu=0.1, gamma=0.1, theta=0.1, delta=0.1,
                 max_iter_outer=8, max_iter_inner_p=3):
        self.n_components = n_components
        self.mu = mu
        self.gamma = gamma
        self.theta = theta
        self.delta = delta
        self.max_iter_outer = max_iter_outer
        self.max_iter_inner_p = max_iter_inner_p
        self.W = None

    def fit(self, S_w_list, S_b_list, n_features):
        m_views = len(S_w_list)
        alpha = np.ones(m_views) / m_views
        beta = np.ones(m_views) / m_views
        P = np.eye(n_features, self.n_components)

        for out_it in range(self.max_iter_outer):

            M_base = np.zeros((n_features, n_features))
            for i in range(m_views):
                M_base += alpha[i] * S_w_list[i] - self.mu * beta[i] * S_b_list[i]


            for in_it in range(self.max_iter_inner_p):

                row_norms = np.linalg.norm(P, axis=1)
                d_vec = 1.0 / (2.0 * row_norms + 1e-10)
                D = np.diag(d_vec)

                M_total = M_base + self.gamma * D
                eig_vals, eig_vecs = la.eigh(M_total)
                P = eig_vecs[:, :self.n_components]


            w_vals = np.array([np.trace(P.T @ Sw @ P) for Sw in S_w_list])
            alpha = self._update_weights(w_vals, self.theta, mode='min')

            b_vals = np.array([np.trace(P.T @ Sb @ P) for Sb in S_b_list])
            beta = self._update_weights(b_vals, self.delta, mode='max')

            # 在 fit 循环末尾
            #if out_it == self.max_iter_outer - 1:
            #    print(f"Final Alpha (Sw weights): {alpha}")
            #    print(f"Final Beta (Sb weights): {beta}")

        self.W = P
        return self

    def _update_weights(self, vals, param, mode='min'):
        m = len(vals)
        if mode == 'min':
            raw = 1.0 / m + (np.mean(vals) - vals) / (2 * param + 1e-10)
        else:
            raw = 1.0 / m + (vals - np.mean(vals)) / (2 * param + 1e-10)
        w = np.maximum(raw, 0)
        return w / (np.sum(w) + 1e-10)

    def transform(self, X):
        return X @ self.W


def compute_scatters_fast(X, y, k_list, t_threshold):
    n_samples, n_features = X.shape
    classes = np.unique(y)
    S_w_list, S_b_list = [], []

    for k in k_list:
        rnn_map = {i: [] for i in range(n_samples)}
        for c in classes:
            idx_c = np.where(y == c)[0]
            if len(idx_c) <= k: continue
            nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X[idx_c])
            indices = nbrs.kneighbors(X[idx_c], return_distance=False)
            for l_q, neighbors in enumerate(indices):
                q = idx_c[l_q]
                for l_p in neighbors[1:]:
                    p = idx_c[l_p]
                    rnn_map[p].append(q)

        valid_idx = [i for i in range(n_samples) if len(rnn_map[i]) >= t_threshold]
        Sw = np.zeros((n_features, n_features))
        Sb = np.zeros((n_features, n_features))

        if valid_idx:
            m_tildes = np.array([np.mean(X[rnn_map[i]], axis=0) for i in valid_idx])

            for idx, i in enumerate(valid_idx):
                diff_w = X[rnn_map[i]] - m_tildes[idx]
                Sw += diff_w.T @ diff_w


            y_valid = y[valid_idx]
            for idx in range(len(valid_idx)):
                diff_b = m_tildes[idx] - m_tildes[idx + 1:]
                mask = y_valid[idx] != y_valid[idx + 1:]
                if np.any(mask):
                    relevant_diffs = diff_b[mask]
                    Sb += relevant_diffs.T @ relevant_diffs

            if np.trace(Sw) > 0: Sw /= np.trace(Sw)
            if np.trace(Sb) > 0: Sb /= np.trace(Sb)

        S_w_list.append(Sw)
        S_b_list.append(Sb)
    return S_w_list, S_b_list


def load_data(path):
    print(f"正在加载数据: {path}")
    data = []
    with open(path, 'r') as f:
        start_read = False
        for line in f:
            line = line.strip()
            if not line or line.startswith('@'):
                if line.lower().startswith('@data'):
                    start_read = True
                continue
            if start_read:
                parts = line.split(',')
                data.append(parts)

    df = pd.DataFrame(data)
    X = df.iloc[:, :-1].values.astype(np.float64)
    y_raw = df.iloc[:, -1].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    return X, y


if __name__ == "__main__":
    data_path =
    random_seeds =range()


    dimensions =

    mu_list =
    gamma_list =
    theta_list =
    delta_list =

    k_list =[2,4,6]

    X_all, y_all = load_data(data_path)




    seed_data_cache = []
    for seed in random_seeds:
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.3, random_state=seed, stratify=y_all
        )
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train_raw)
        X_te_s = scaler.transform(X_test_raw)
        X_tr_pca = X_tr_s
        X_te_pca = X_te_s
        #pca = PCA(n_components=0.95).fit(X_tr_s)
        #X_tr_pca = pca.transform(X_tr_s)
        #X_te_pca = pca.transform(X_te_s)
        Sw_l, Sb_l = compute_scatters_fast(X_tr_pca, y_train, k_list, 10)
        seed_data_cache.append((X_tr_pca, X_te_pca, y_train, y_test, Sw_l, Sb_l))


    header = f"{'Dim':<4} | {'mu':<5} | {'gam':<5} | {'the':<5} | {'del':<5} | {'Accuracy (Mean ± Std)':<22} | {'Time':<10}"
    print(header)
    print("-" * len(header))

    results_log = []
    for d in dimensions:
        for mu in mu_list:
            for gamma in gamma_list:
                for theta in theta_list:
                    for delta in delta_list:
                        start_time = time.time()
                        acc_list = []


                        for X_tr, X_te, y_tr, y_te, Sw_l, Sb_l in seed_data_cache:
                            d_act = min(d, X_tr.shape[1])


                            model = RNDA(
                                n_components=d_act,
                                mu=mu,
                                gamma=gamma,
                                theta=theta,
                                delta=delta
                            )
                            model.fit(Sw_l, Sb_l, X_tr.shape[1])

                            X_tr_red = model.transform(X_tr)
                            X_te_red = model.transform(X_te)

                            knn = KNeighborsClassifier(n_neighbors=1).fit(X_tr_red, y_tr)
                            acc_list.append(accuracy_score(y_te, knn.predict(X_te_red)))

                        mean_acc = np.mean(acc_list)
                        std_acc = np.std(acc_list)
                        duration = time.time() - start_time

                        avg_duration = duration / len(random_seeds)


                        print(
                            f"{d:<4} | {mu:<5.2f} | {gamma:<5.2f} | {theta:<5.2f} | {delta:<5.2f} | {mean_acc:.4f} ± {std_acc:.4f} | {avg_duration:.5f}s")

                        results_log.append({
                            'dim': d, 'mu': mu, 'gamma': gamma,
                            'theta': theta, 'delta': delta,
                            'mean': mean_acc, 'std': std_acc
                        })


    best = max(results_log, key=lambda x: x['mean'])
    print("\n" + "=" * 70)

    print(f"Best (Dim): {best['dim']}")
    print(f"Best Mu:    {best['mu']}")
    print(f"Best Gamma: {best['gamma']}")
    print(f"Best Theta: {best['theta']}")
    print(f"Best Delta: {best['delta']}")
    print(f"Best Accuracy: {best['mean']:.4f} ± {best['std']:.4f}")
    print("=" * 70)