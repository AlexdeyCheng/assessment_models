'''
Docstring for AHP.py

It's a python file running AHP algorithm based on package pyDecision.
Further information about AHP_method can be found in AHP.ipynb file under the same folder.

'''

from pyDecision.algorithm import ahp_method
import numpy as np

# =========================
# Parameters
# =========================
weight_derivation = 'geometric'   # 'mean' | 'geometric' | 'max_eigen'
rc_threshold = 0.10

# =========================
# Criteria layer (n criteria): pairwise comparison matrix
# =========================
criteria = np.array(["g1", "g2", "g3", "g4", "g5", "g6", "g7"])
n = len(criteria)

criteria_pcm = np.array([
    # g1     g2     g3     g4     g5     g6     g7
    [1  ,   1/3,   1/5,   1  ,   1/4,   1/2,   3  ],   # g1
    [3  ,   1  ,   1/2,   2  ,   1/3,   3  ,   3  ],   # g2
    [5  ,   2  ,   1  ,   4  ,   5  ,   6  ,   5  ],   # g3
    [1  ,   1/2,   1/4,   1  ,   1/4,   1  ,   2  ],   # g4
    [4  ,   3  ,   1/5,   4  ,   1  ,   3  ,   2  ],   # g5
    [2  ,   1/3,   1/6,   1  ,   1/3,   1  ,   1/3],   # g6
    [1/3,   1/3,   1/5,   1/2,   1/2,   3  ,   1  ]    # g7
], dtype=float)

# =========================
# Alternatives layer (m alternatives): define explicitly (no randomness)
# =========================
alternatives = np.array(["a1", "a2", "a3", "a4"])
m = len(alternatives)

# 对每个准则，显式给出一个 m×m 的方案成对比较矩阵（互反矩阵，主对角为1）
# 你后续只需要直接修改下面这些矩阵的数值即可。
alternatives_pcm_list = [
    # ---- 对准则 g1 的方案成对比较矩阵 ----
    np.array([
        [1,    3,    5,    7],
        [1/3,  1,    3,    5],
        [1/5,  1/3,  1,    3],
        [1/7,  1/5,  1/3,  1],
    ], dtype=float),

    # ---- 对准则 g2 ----
    np.array([
        [1,    1/2,  4,    6],
        [2,    1,    5,    7],
        [1/4,  1/5,  1,    3],
        [1/6,  1/7,  1/3,  1],
    ], dtype=float),

    # ---- 对准则 g3 ----
    np.array([
        [1,    1/4,  1/6,  1/3],
        [4,    1,    1/3,  2],
        [6,    3,    1,    4],
        [3,    1/2,  1/4,  1],
    ], dtype=float),

    # ---- 对准则 g4 ----
    np.array([
        [1,    2,    7,    5],
        [1/2,  1,    5,    3],
        [1/7,  1/5,  1,    1/2],
        [1/5,  1/3,  2,    1],
    ], dtype=float),

    # ---- 对准则 g5 ----
    np.array([
        [1,    1/3,  3,    4],
        [3,    1,    5,    6],
        [1/3,  1/5,  1,    2],
        [1/4,  1/6,  1/2,  1],
    ], dtype=float),

    # ---- 对准则 g6 ----
    np.array([
        [1,    4,    2,    6],
        [1/4,  1,    1/2,  2],
        [1/2,  2,    1,    3],
        [1/6,  1/2,  1/3,  1],
    ], dtype=float),

    # ---- 对准则 g7 ----
    np.array([
        [1,    3,    1/2,  2],
        [1/3,  1,    1/5,  1/2],
        [2,    5,    1,    4],
        [1/2,  2,    1/4,  1],
    ], dtype=float),
]

# 可选：简单检查互反性（不影响主流程）
def _check_reciprocal(pcm: np.ndarray, tol: float = 1e-8) -> bool:
    if pcm.shape[0] != pcm.shape[1]:
        return False
    if not np.allclose(np.diag(pcm), 1.0, atol=tol):
        return False
    return np.allclose(pcm * pcm.T, np.ones_like(pcm), atol=tol)

for k in range(n):
    if not _check_reciprocal(alternatives_pcm_list[k]):
        raise ValueError(f"alternatives_pcm_list[{k}] (for criterion {criteria[k]}) is not reciprocal / diagonal not 1.")

# =========================
# Run AHP: criteria weights + criteria consistency
# =========================
criteria_weights, criteria_rc = ahp_method(criteria_pcm, wd=weight_derivation)
criteria_rc = float(criteria_rc)

# =========================
# Run AHP: alternatives weights under each criterion + consistency
# Build W_alt (m×n)
# =========================
W_alt = np.zeros((m, n), dtype=float)
alternatives_rc_list = []

for k in range(n):
    w_k, rc_k = ahp_method(alternatives_pcm_list[k], wd=weight_derivation)
    W_alt[:, k] = w_k
    alternatives_rc_list.append(float(rc_k))

# =========================
# Final total scores: score = W_alt (m×n) @ w_criteria (n×1)
# =========================
score = (W_alt @ criteria_weights.reshape(-1, 1)).reshape(-1)  # (m,)

# =========================
# Output (per your required format)
# =========================
print("准则层：")
for i, name in enumerate(criteria, start=1):
    print(f"    {i}. w({name}) = {criteria_weights[i-1]:.6f}")
print(f"    rc = {criteria_rc:.3f}")
print("    一致性通过" if criteria_rc <= rc_threshold else "    一致性未通过")

print("方案层：")
for k, crit_name in enumerate(criteria, start=1):
    print(f"对准则 {crit_name}:")
    for j, alt_name in enumerate(alternatives, start=1):
        print(f"    {j}. w({alt_name}) = {W_alt[j-1, k-1]:.6f}")
    rc_k = alternatives_rc_list[k-1]
    print(f"    rc = {rc_k:.3f}")
    print("    一致性通过" if rc_k <= rc_threshold else "    一致性未通过")

print("总得分：")
for j, alt_name in enumerate(alternatives, start=1):
    print(f"    {j}. {alt_name} = {score[j-1]:.6f}")

# ======================================================================================
# （注释块）方式B：将一个 m×n 的原始矩阵（方案×准则）转换为“对每个准则一个 m×m 成对比较矩阵”
# 注意：变量接口与上文一致：criteria、alternatives、m、n；输出可赋给 alternatives_pcm_list
# ======================================================================================
# raw_matrix = np.array([
#     # c1   c2   c3  ... (n columns)
#     # [x11, x12, x13, ...],
#     # [x21, x22, x23, ...],
#     # ...
# ], dtype=float)  # shape: (m, n) = (len(alternatives), len(criteria))
#
# # -------- 方法1：比例法（ratio）--------
# # 适用于“越大越好”的效益型指标（benefit）。若是成本型（cost，越小越好），可先用 col = 1/col 或做反向变换。
# # 会生成严格互反矩阵：pcm[i,j] = col[i]/col[j]
# alternatives_pcm_list = []
# for k in range(n):
#     col = raw_matrix[:, k].astype(float)
#     col = np.where(col <= 0, 1e-12, col)  # 避免0/负数
#     pcm_k = col[:, None] / col[None, :]
#     alternatives_pcm_list.append(pcm_k)
#
# # -------- 方法2：差值映射到 Saaty 1~9 标度（difference -> 1..9）--------
# # 思路：对每列 col，取两两差值，按差值占最大差值的比例映射为 1..9 强度，再保持互反。
# # 适合 raw_matrix 是“原始分值/测量值”，你希望得到更接近传统AHP的 1..9 成对比较。
# # （若是成本型指标，可把 d = col[i] - col[j] 的符号逻辑反过来，或先对 col 做反向变换）
# #
# # def _map_intensity(abs_d: float, max_abs_d: float) -> int:
# #     if max_abs_d <= 0:
# #         return 1
# #     intensity = int(np.round(1 + 8 * (abs_d / max_abs_d)))
# #     return max(1, min(9, intensity))
# #
# # alternatives_pcm_list = []
# # for k in range(n):
# #     col = raw_matrix[:, k].astype(float)
# #     max_abs_d = np.max(np.abs(col[:, None] - col[None, :]))
# #     pcm_k = np.ones((m, m), dtype=float)
# #     for i in range(m):
# #         for j in range(i + 1, m):
# #             d = col[i] - col[j]
# #             intensity = _map_intensity(abs(d), max_abs_d)
# #             val = float(intensity) if d >= 0 else 1.0 / float(intensity)
# #             pcm_k[i, j] = val
# #             pcm_k[j, i] = 1.0 / val
# #     alternatives_pcm_list.append(pcm_k)
#
# # 接下来你可直接复用主流程，对每个 alternatives_pcm_list[k] 调用 ahp_method(...)
