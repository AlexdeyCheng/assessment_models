import numpy as np
import warnings
import skfuzzy as fuzz
from cycler import cycler
import matplotlib.pyplot as plt
import itertools

# =============================================================================
# 1. Configuration Parser (配置解析器)
# =============================================================================

def _parse_criteria_config(criteria_type_input):
    """
    [Internal] Parse user's mixed input list into a standardized list of dictionaries.
    [内部函数] 将用户的混合输入列表解析为标准化的字典列表。

    Design Philosophy:
    - Frontend: Permissive (supports strings and lists).
    - Backend: Strict (only processes dictionaries).
    
    设计哲学：
    - 前端：宽容（支持字符串和列表混写）。
    - 后端：严谨（只处理字典结构）。

    Args:
        criteria_type_input (list): e.g., ['max', ['interval', 3, 5], ['nominal', 7]]

    Returns:
        list[dict]: e.g., [{'type': 'max', 'params': {}}, {'type': 'interval', 'params': {'a': 3, 'b': 5}}]
    """
    standardized_list = []
    
    # Validation: Input must be iterable
    if not isinstance(criteria_type_input, (list, tuple, np.ndarray)):
        raise TypeError("criteria_type_input must be a list or array.")

    for idx, item in enumerate(criteria_type_input):
        # Case A: Simple String (e.g., 'max', 'min')
        if isinstance(item, str):
            c_type = item.lower().strip()
            if c_type not in ['max', 'min']:
                raise ValueError(f"Index {idx}: Unknown type string '{item}'. Supported: 'max', 'min'.")
            standardized_list.append({'type': c_type, 'params': {}})
            
        # Case B: List/Tuple with Params (e.g., ['interval', 3, 5])
        elif isinstance(item, (list, tuple, np.ndarray)):
            if len(item) == 0:
                raise ValueError(f"Index {idx}: Empty configuration found.")
            
            c_type = str(item[0]).lower().strip()
            params = item[1:] # Extract parameters
            p_dict = {}
            
            if c_type == 'nominal':
                if len(params) != 1:
                    raise ValueError(f"Index {idx} ('nominal'): Requires exactly 1 target value. Got {params}.")
                p_dict['target'] = float(params[0])
                
            elif c_type == 'interval':
                if len(params) != 2:
                    raise ValueError(f"Index {idx} ('interval'): Requires 2 values [a, b]. Got {params}.")
                # Auto-sort to ensure a <= b
                p_dict['a'] = min(params)
                p_dict['b'] = max(params)
                
            else:
                raise ValueError(f"Index {idx}: Unknown complex type '{c_type}'. Supported: 'nominal', 'interval'.")
            
            standardized_list.append({'type': c_type, 'params': p_dict})
            
        else:
            raise TypeError(f"Index {idx}: Invalid format. Expected str or list, got {type(item)}.")
            
    return standardized_list


# =============================================================================
# 2. Generic Alignment Checker (通用对齐检查工具)
# =============================================================================

def check_alignment(matrix, vector_obj, axis=1):
    """
    [Math Tool] Check if a matrix dimension matches a vector length.
    [通用数学工具] 检查矩阵的某一维是否与向量长度匹配。
    
    Reuse: This function is algorithm-agnostic. Can be used for PCA, TOPSIS, FCE, etc.

    Args:
        matrix (np.ndarray): The data matrix.
        vector_obj (list/array): The vector to check against.
        axis (int): The axis of matrix to compare (0 for rows, 1 for columns).

    Returns:
        tuple: (is_aligned (bool), matrix_dim (int), vector_len (int))
    """
    mat = np.asarray(matrix)
    
    if mat.ndim < 2:
        return False, mat.shape[0], len(vector_obj)
        
    m_dim = mat.shape[axis]
    v_len = len(vector_obj)
    
    is_aligned = (m_dim == v_len)
    
    return is_aligned, m_dim, v_len


# =============================================================================
# 3. FCE Input Preparation (FCE 输入准备与校验)
# =============================================================================

def prepare_inputs(ori_matrices, criteria_type, criteria_name=None, weights=None):
    """
    [Business Logic] Validate and standardize inputs specifically for FCE/TOPSIS.
    [业务逻辑] 专门为 FCE/TOPSIS 准备和校验输入数据。

    Logic:
    1. Type Check: STRICT. Must match matrix columns.
    2. Name Check: WEAK. If mismatch, auto-generate (c1, c2...).
    3. Weight Check: NUMERIC. Must match length and sum to 1.
    
    Args:
        ori_matrices (array-like): Raw data (n_samples, m_criteria).
        criteria_type (list): The mixed type list.
        criteria_name (list, optional): Names of criteria.
        weights (list, optional): Weights of criteria.

    Returns:
        dict: A clean dictionary containing all necessary components.
              {'X': array, 'types': list[dict], 'names': array, 'weights': array, 'm': int, 'n': int}
    """
    # 1. Standardize Matrix
    X = np.asarray(ori_matrices, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Input matrix must be 2D (samples, criteria). Got shape {X.shape}")
    
    n_samples, m_criteria = X.shape
    
    # 2. Process & Validate Types (STRICT)
    parsed_types = _parse_criteria_config(criteria_type)
    is_aligned, m_dim, t_len = check_alignment(X, parsed_types, axis=1)
    
    if not is_aligned:
        raise ValueError(
            f"CRITICAL ERROR: Criteria Type mismatch.\n"
            f"Matrix has {m_dim} columns (criteria), but you provided {t_len} types.\n"
            f"These MUST match exactly."
        )

    # 3. Process Names (WEAK/Auto-fill)
    final_names = []
    
    if criteria_name is None:
        final_names = np.array([f"c{i+1}" for i in range(m_criteria)])
    else:
        is_aligned, _, n_len = check_alignment(X, criteria_name, axis=1)
        if is_aligned:
            final_names = np.asarray(criteria_name)
        else:
            warnings.warn(
                f"Dimension Mismatch in 'criteria_name': Matrix({m_criteria}) vs Names({n_len}). "
                f"Ignoring provided names and auto-generating ['c1', 'c2', ...].",
                UserWarning
            )
            final_names = np.array([f"c{i+1}" for i in range(m_criteria)])

    # 4. Process Weights (Numeric Check)
    final_weights = None
    if weights is not None:
        w_arr = np.asarray(weights, dtype=float)
        
        is_aligned, _, w_len = check_alignment(X, w_arr, axis=1)
        if not is_aligned:
            raise ValueError(
                f"Weight Length Mismatch: Matrix has {m_criteria} criteria, "
                f"but weights vector has length {w_len}."
            )
            
        if not np.isclose(np.sum(w_arr), 1.0, atol=1e-5):
            raise ValueError(f"Weights must sum to 1. Current sum: {np.sum(w_arr):.4f}")
            
        final_weights = w_arr

    # 5. Return Clean Dictionary
    return {
        'X': X,
        'm': m_criteria,
        'n': n_samples,
        'types': parsed_types,
        'names': final_names,
        'weights': final_weights
    }


# =============================================================================
# 4. Core Calculation Functions (核心计算模块)
# =============================================================================

def membership_factory(x, mf_type, params):
    """
    隶属度计算工厂函数 / Membership Calculation Factory.
    支持单维向量 (n_samples,) 或 多维特征矩阵 (n_samples, n_features) 的隶属度计算。
    
    :param x: 输入的数据矩阵或向量 (numpy.ndarray)。
    :param mf_type: 字符串，指定隶属度函数形状 (e.g., 'gaussmf', 'trimf', 'trapmf')。
    :param params: 形状参数。
                   - 计算单集合：[p1, p2, ...] 
                   - 计算多集合(聚类)：[[p1, p2, ...], [p1, p2, ...]]
    :return: 隶属度矩阵 U。
    """
    MF_MAP = {
        'trimf': fuzz.trimf,
        'gaussmf': fuzz.gaussmf,
        'trapmf': fuzz.trapmf,
        'gbellmf': fuzz.gbellmf,
        'sigmf': fuzz.sigmf
    }

    if mf_type not in MF_MAP:
        raise ValueError(f"Unsupported membership function type: {mf_type}")

    func = MF_MAP[mf_type]
    
    # 定义需要解包参数的函数名单
    needs_unpacking = ['gaussmf', 'gbellmf', 'sigmf']

    def _adapter(data, p):
        # 嵌套定义：直接读取外部变量 mf_type 和 func 
        if mf_type in needs_unpacking:
            return func(data, *p)
        return func(data, p)

    # 自动识别是计算单集合(vector)还是多集合(matrix)
    if isinstance(params[0], (list, np.ndarray)):
        return np.array([_adapter(x, p) for p in params])
    
    return _adapter(x, params)


def cmeans_membership_update(distance_matrix, m=2.0):
    """
    基于距离矩阵自动更新隶属度 (模糊聚类核心后验逻辑)。
    依据公式: u_ij = 1 / sum( (d_ij / d_kj) ^ (2 / (m-1)) )
    """
    # 防止除以零
    eps = np.finfo(np.float64).eps
    distance_matrix = np.fmax(distance_matrix, eps)    

    # 计算距离的幂次项
    power = 2.0 / (m - 1.0)
    temp = distance_matrix ** (-power)
    
    # 归一化
    u_matrix = temp / temp.sum(axis=0)
    
    return u_matrix


def fuzzy_synthesis(R, W, operator='product_sum'):
    """
    模糊综合评价核心合成算法 (Fuzzy Composition).
    执行 B = W ∘ R 运算。

    Args:
        R (np.ndarray): 模糊关系矩阵 (Membership Matrix), (m_criteria, k_grades).
        W (np.ndarray): 权重向量 (Weight Vector), (m_criteria,).
        operator (str): 模糊合成算子类型。
                        - 'product_sum': M(·, +) 常规加权平均 (默认).
                        - 'min_max':     M(∧, ∨) 扎德算子 (Zadeh).
                        - 'product_max': M(·, ∨) 乘积最大型.

    Returns:
        np.ndarray: 综合评价结果向量 B, (k_grades,).
    """
    R = np.asarray(R)
    W = np.asarray(W)
    
    if W.ndim != 1:
        W = W.flatten()
        
    m_criteria = W.shape[0]
    
    if R.shape[0] != m_criteria:
        raise ValueError(
            f"Dimension Mismatch: Weights length is {m_criteria}, "
            f"but Membership Matrix R has {R.shape[0]} rows (criteria)."
        )

    op = operator.lower().strip()

    # --- 模型 I: M(·, +) 加权平均型 ---
    if op in ['product_sum', 'dot', 'mean']:
        B = np.dot(W, R)

    # --- 模型 II: M(∧, ∨) 主因素决定型 (Zadeh) ---
    elif op in ['min_max', 'zadeh']:
        weighted_R = np.minimum(W[:, None], R)
        B = np.max(weighted_R, axis=0)

    # --- 模型 III: M(·, ∨) 乘积最大型 ---
    elif op in ['product_max']:
        weighted_R = W[:, None] * R
        B = np.max(weighted_R, axis=0)

    else:
        raise ValueError(f"Unsupported fuzzy operator: '{op}'.")

    # 结果归一化
    total = np.sum(B)
    if not np.isclose(total, 0.0):
        B_normalized = B / total
    else:
        B_normalized = B

    return B_normalized


def defuzzify(B, method='max_membership', scores=None):
    """
    解模糊/去模糊化：将模糊向量转化为单值结果。

    Args:
        B (np.ndarray): 综合评价结果向量.
        method (str): 'max_membership' or 'weighted_score'.
        scores (list): Grade scores for weighted method.

    Returns:
        (index, value) or float score.
    """
    if method == 'max_membership':
        idx = np.argmax(B)
        val = B[idx]
        return idx, val
    
    elif method == 'weighted_score':
        if scores is None:
            raise ValueError("Must provide 'scores' list for weighted_score method.")
        if len(scores) != len(B):
            raise ValueError(f"Scores length ({len(scores)}) does not match vector B length ({len(B)}).")
        return np.dot(B, np.array(scores))
    
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# 5. Internal Helpers: Preprocessing & Parameter Generation (内部工具：预处理与参数生成)
# =============================================================================

def _preprocess_column(col_data, c_config):
    """
    [Internal] 单列数据预处理函数 / Single Column Preprocessing.
    
    Logic:
    - Standardize ALL types (max, min, nominal, interval) into a [0, 1] proximity score.
    - Higher score ALWAYS means better performance (Benefit type / Max type).
    
    逻辑：
    - 将所有类型（极大、极小、点值、区间）统一归一化为 [0, 1] 的得分。
    - 得分越高，代表表现越好（统一为效益型/Max型）。
    
    Args:
        col_data (np.array): 1D array of column data.
        c_config (dict): Configuration dictionary.
        
    Returns:
        tuple: (processed_data, effective_config)
    """
    c_type_str = c_config['type']
    
    # 获取列数据的极值 (用于归一化)
    d_min, d_max = np.min(col_data), np.max(col_data)
    
    # 防止极差为0 (单一点数据) / Avoid division by zero
    if np.isclose(d_min, d_max):
        # 这种情况下无法比较，直接全给满分
        return np.ones_like(col_data), {'type': 'max'}

    # -------------------------------------------------------
    # Case 1: Max (效益型 -> 线性归一化)
    # -------------------------------------------------------
    if c_type_str == 'max':
        # 公式: (x - min) / (max - min)
        score = (col_data - d_min) / (d_max - d_min)
        return score, {'type': 'max'}

    # -------------------------------------------------------
    # Case 2: Min (成本型 -> 线性归一化)
    # -------------------------------------------------------
    elif c_type_str == 'min':
        # === [预埋功能：倒数转化法] (目前 Bypass / Commented Out) ===
        # 说明：某些文献使用 1/x 进行转化，但会导致数据尺度非线性且不在 [0,1] 间。
        # use_inverse = False 
        # if use_inverse:
        #     # 防止除0
        #     safe_data = np.where(col_data == 0, 1e-9, col_data)
        #     score = 1.0 / safe_data
        #     return score, {'type': 'max'}
        
        # === [当前逻辑：线性归一化 / Linear Normalization] ===
        # 公式: (max - x) / (max - min)
        # x 越小 -> 分子越大 -> 得分越高
        score = (d_max - col_data) / (d_max - d_min)
        return score, {'type': 'max'}

    # -------------------------------------------------------
    # Case 3: Nominal (定性/点值 -> 距离得分)
    # -------------------------------------------------------
    elif c_type_str == 'nominal':
        target = c_config['params']['target']
        # 1. 计算距离
        dist = np.abs(col_data - target)
        max_dist = np.max(dist)
        
        if np.isclose(max_dist, 0):
            score = np.ones_like(dist)
        else:
            # 2. 正向化公式: S = 1 - d / d_max
            score = 1.0 - (dist / max_dist)
            
        return score, {'type': 'max'}
        
    # -------------------------------------------------------
    # Case 4: Interval (区间 -> 距离得分)
    # -------------------------------------------------------
    elif c_type_str == 'interval':
        a = c_config['params']['a']
        b = c_config['params']['b']
        
        dist = np.zeros_like(col_data)
        mask_low = col_data < a
        mask_high = col_data > b
        
        # 计算距离
        dist[mask_low] = a - col_data[mask_low]
        dist[mask_high] = col_data[mask_high] - b
        
        max_dist = np.max(dist)
        
        if np.isclose(max_dist, 0):
            score = np.ones_like(dist)
        else:
            # 正向化公式: S = 1 - d / d_max
            score = 1.0 - (dist / max_dist)
            
        return score, {'type': 'max'}
    
    else:
        # 未知类型，理论上在上层 parser 就会被拦截
        raise ValueError(f"Unknown criteria type: {c_type_str}")


def _auto_generate_params(data_vector, c_type_config, k_grades):
    """
    [Internal] Generate membership parameters based on data distribution.
    [内部函数] 根据数据生成 'trimf' 参数。
    
    Logic:
        - Assumes input `data_vector` is ALREADY normalized to a [0, 1] benefit score.
        - Therefore, Best Grade is ALWAYS at max(data), Worst Grade at min(data).
        
    逻辑：
    - 假设输入数据已经被预处理为“越大越好”的得分（通常在 [0, 1] 之间）。
    - 因此，最优等级永远定锚在最大值，最差等级在最小值。
    
    Returns:
        mf_type (str): 'trimf'
        params (list): List of parameter sets.
    """
    # 1. 获取极值 (对于归一化数据，理论上是 0 和 1，但取实际极值更稳健)
    d_min, d_max = np.min(data_vector), np.max(data_vector)
    
    # 防止极差为0 / Prevent division by zero error
    if np.isclose(d_min, d_max):
        d_min -= 0.01
        d_max += 0.01

    # 2. 生成中心点 (从大到小：Best -> Worst)
    # Note: 不再需要判断 'min'/'cost'，因为在 _preprocess_column 中已经转化为了 'max'
    centers = np.linspace(d_max, d_min, k_grades)
    
    # 3. 根据中心点生成三角形参数 [left, peak, right]
    params = []
    
    for i in range(k_grades):
        peak = centers[i]
        
        # 确定左脚 (Left)
        if i == 0:
            # G1 (Best): 左脚重合或延伸 / First grade starts at peak
            left = peak 
        else:
            left = centers[i-1] # 前一个中心
            
        # 确定右脚 (Right)
        if i == k_grades - 1:
            # Gk (Worst): 右脚重合或延伸 / Last grade ends at peak
            right = peak
        else:
            right = centers[i+1] # 后一个中心
            
        # 排序确保符合 trimf 格式 [a, b, c]
        # 因为 centers 是降序的，直接由 [left, peak, right] 组成可能会乱序，必须 sort
        tri_params = sorted([left, peak, right])
        params.append(tri_params)
        
    return 'trimf', params


# =============================================================================
# 6. Visualization Functions (可视化功能模块)
# =============================================================================

def default_plot_sets(theme='blue_grey'):
    """
    配置全局绘图风格并返回主题颜色配置。
    
    1. Sets global rcParams (Arial font, High DPI).
    2. Returns a dictionary containing 'cmap' (for heatmap) and 'colors' (for radar).
    """
    # --- 1. Global Style Settings ---
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "Arial",        # [Req 1] 统一为 Arial 字体
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.5,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # --- 2. Theme Definitions ---
    themes = {
        'blue_grey': {
            'cmap': 'Blues',
            # Deep Blue, Blue Grey, Light Blue, Steel Blue, Slate Grey
            'colors': ['#2E5B88', '#7A8B99', '#B0C4DE', '#4682B4', '#708090']
        },
        'green_purple': {
            'cmap': 'viridis', 
            # Purple, Teal, Green, Light Green, Dark Purple
            'colors': ['#482677', '#2D708E', '#20A387', '#73D055', '#440154']
        },
        'grey_yellow': {
            'cmap': 'cividis', 
            # Dark Grey, Gold, Grey, Dark Golden Rod, Black
            'colors': ['#404040', '#D4AF37', '#808080', '#B8860B', '#000000']
        }
    }
    
    # Return selected theme config (Fallback to blue_grey)
    return themes.get(theme, themes['blue_grey'])


def plot_fuzzy_heatmap(B_matrix, sample_names, grade_names, theme_config, save_prefix=None):
    """
    绘制结果矩阵热力图 (Samples x Grades)。
    """
    cmap = theme_config['cmap']
    n_samples, n_grades = B_matrix.shape
    
    # 动态调整高度：每个样本预留一定高度，防止重叠
    fig_height = max(4, n_samples * 0.5 + 1.5)
    fig, ax = plt.subplots(figsize=(6, fig_height))
    
    # Draw Heatmap
    im = ax.imshow(B_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # Add Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Membership Degree", rotation=-90, va="bottom")

    # Set Ticks
    ax.set_xticks(np.arange(n_grades))
    ax.set_yticks(np.arange(n_samples))
    
    # Set Labels
    ax.set_xticklabels(grade_names)
    ax.set_yticklabels(sample_names)
    
    # Style Ticks
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Add Text Annotations (Values)
    threshold = 0.5 
    for i in range(n_samples):
        for j in range(n_grades):
            val = B_matrix[i, j]
            color = "white" if val > threshold else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    ax.set_title("Fuzzy Evaluation Results")
    fig.tight_layout()

    # Save Logic
    if save_prefix:
        filename = f"{save_prefix}_heatmap.jpg"
        plt.savefig(filename, bbox_inches='tight')
        print(f"[Plot] Heatmap saved to: {filename}")
    
    plt.show()


def plot_fuzzy_radar(B_matrix, sample_names, grade_names, theme_config, save_prefix=None):
    """
    绘制雷达图 (修复版 V2：增加标记点和线宽，确保极端数据可见)
    """
    colors = theme_config['colors']
    num_vars = len(grade_names)
    n_samples = len(sample_names)
    
    # 1. 角度计算
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    # 2. 画布设置
    fig, ax = plt.subplots(figsize=(8, 6), subplot_kw=dict(polar=True))
    
    # 3. 颜色循环
    color_cycle = itertools.cycle(colors)
    
    # 4. 绘图循环
    for idx, (row, color) in enumerate(zip(B_matrix, color_cycle)):
        if idx >= len(sample_names):
            break
            
        s_name = sample_names[idx]
        values = row.tolist()
        values += values[:1]
        
        # --- 关键修改 ---
        # 增加 marker='o' (圆点标记), markersize=5, linewidth=2
        ax.plot(angles, values, linewidth=2, label=s_name, color=color, marker='o', markersize=5)
        # 稍微降低填充透明度，让线条更明显
        ax.fill(angles, values, color=color, alpha=0.05) 

    # 5. 坐标轴设置
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(grade_names)
    
    ax.set_ylim(0, 1)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="grey", size=8)
    
    ax.set_title("Evaluation Distribution Profile", y=1.08)
    
    # 6. 图例位置
    ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1.0), borderaxespad=0., frameon=False)
    
    # 7. 布局调整
    plt.tight_layout()
    
    if save_prefix:
        filename = f"{save_prefix}_radar.jpg"
        plt.savefig(filename, bbox_inches='tight')
        print(f"[Plot] Radar chart saved to: {filename}")
        
    plt.show()


