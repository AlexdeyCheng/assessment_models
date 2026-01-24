import fce_package as fce
import numpy as np
import warnings


def fuzzy_comprehensive_method(
    criterion_names, 
    criterion_types, 
    grade_names,
    ori_matrices, 
    weights=None,      # Default: None (will be equal weights)
    sample_names=None, # Custom sample names or auto-generated
    
    # Optional: Graphs
    theme='blue_grey', # Plot theme selection, defaults to first option
    save_prefix=None,  # Save prefix for plots <--- 【Fixed: Added comma】
    graph_heat=False, 
    graph_radar=False,
):
    """
    模糊综合评价 (FCE) 主调度函数。
    
    Process:
    1. ETL: Clean and validate inputs.
    2. Auto-Binning: Generate parameters based on data stats.
    3. Calculation: Compute Membership Matrix (R) and Result Vector (B).
    4. Visualization: (Optional) Plot Heatmap/Radar.
    
    Args:
        criterion_names (list): Names of criteria (e.g., ['Price', 'Quality']).
        criterion_types (list): Types (e.g., ['min', 'max', ['nominal', 5]]).
        grade_names (list): Names of grades (e.g., ['Excellent', 'Good', 'Bad']).
        ori_matrices (np.ndarray): Data matrix (n_samples, m_criteria).
        weights (list, optional): Weights summing to 1. Defaults to equal weights.
        graph_heat (bool): Draw heatmap if True.
        graph_radar (bool): Draw radar chart if True.
        save_prefix (str, optional): Filename prefix for saving plots.

    Returns:
        dict: {
            'Membership_Tensor': (n, m, k) ndarray,  # 完整的三维隶属度数据
            'Results_Matrix': (n, k) ndarray,        # 每个样本的评价向量
            'Defuzzified_Results': list,             # 每个样本的最终等级/得分
            'Raw_Data': dict                         # 清洗后的原始输入
        }
    """
    # ---------------------------------------------------------
    # Step 1: Data Preparation (ETL)
    # ---------------------------------------------------------
    data_dict = fce.prepare_inputs(ori_matrices, criterion_types, criterion_names, weights)
    print("Data Preparation:")
    print(data_dict)

    X = data_dict['X']          # (n_samples, m_criteria)
    types = data_dict['types']  # Standardized type configs
    final_weights = data_dict['weights']
    
    n_samples = data_dict['n']
    m_criteria = data_dict['m']
    k_grades = len(grade_names)
    
    # Handle Weights Default (Equal Weights)
    if final_weights is None:
        final_weights = np.ones(m_criteria) / m_criteria

    # ---------------------------------------------------------
    # Step 2: Calculate Membership Tensor R (n, m, k)
    # ---------------------------------------------------------
    # R_tensor[i, j, :] represents sample i, criteria j, membership vector to grades
    R_list_per_criteria = [] 

    for j in range(m_criteria):
        col_data = X[:, j] # (n,)
        c_config = types[j]
        
        # --- Pre-processing for Complex Types (Refactored) ---
        # 核心逻辑：将 Nominal/Interval 转换为 "Proximity Score" (0-1, 越大越好)
        # 调用新提取的函数，保持主函数整洁
        processing_data, temp_config = fce._preprocess_column(col_data, c_config)

        # --- Auto-Generate Parameters ---
        # 根据处理后的数据(processing_data) 生成几何参数
        mf_type, mf_params = fce._auto_generate_params(processing_data, temp_config, k_grades)
        
        # --- Membership Calculation ---
        # membership_factory 返回 (k_grades, n_samples)
        # 我们需要转置为 (n_samples, k_grades)
        u_matrix_T = fce.membership_factory(processing_data, mf_type, mf_params)
        u_matrix = u_matrix_T.T 
        
        R_list_per_criteria.append(u_matrix)

    # Stack to Tensor: (n_samples, m_criteria, k_grades)
    # axis=1 对应 criteria 维度
    R_tensor = np.stack(R_list_per_criteria, axis=1)

    # ---------------------------------------------------------
    # Step 3: Fuzzy Synthesis (n, m, k) * (m,) -> (n, k)
    # ---------------------------------------------------------
    # 使用 tensordot 进行张量收缩
    # R_tensor axes 1 (m_criteria) 与 final_weights axes 0 (m_criteria) 相乘求和
    # 结果 shape: (n_samples, k_grades)
    
    # TODO：此处是对模糊矩阵的加权平均，即对 tensor[i,j,:] 按 weight[] 加权。
    # fuzzy_synthesis 函数里写好的 min_max (扎德算子) 和 product_max 逻辑，没有被调用
    B_matrix = np.tensordot(R_tensor, final_weights, axes=([1], [0]))
    
    # ---------------------------------------------------------
    # Step 4: Defuzzification
    # ---------------------------------------------------------
    final_results = []
    for i in range(n_samples):
        # 针对每个样本的 B 向量解模糊
        idx, val = fce.defuzzify(B_matrix[i], method='max_membership') 
        
        ##  TODO：添加其他解模糊方法，预埋了 scores 表列作为参数
        final_results.append({
            'sample_index': i,
            'best_grade': grade_names[idx],
            'confidence': val,
            'grade_index': idx
        })

    # ---------------------------------------------------------
    # Step 5: Visualization
    # ---------------------------------------------------------
    
    # 1. 自动补全 sample_names (s1, s2...)
    if sample_names is None:
        sample_names = [f"s{i+1}" for i in range(n_samples)]
    else:
        # 简单校验长度
        if len(sample_names) != n_samples:
             warnings.warn(f"Sample names count ({len(sample_names)}) != n_samples ({n_samples}). Auto-generating default names.")
             sample_names = [f"s{i+1}" for i in range(n_samples)]

    # 2. 执行绘图逻辑
    if graph_heat or graph_radar:
        # 获取主题配置 (包含 cmap 和 colors) 并设置全局字体(Arial)
        theme_config = fce.default_plot_sets(theme)
        
        if graph_heat:
            fce.plot_fuzzy_heatmap(
                B_matrix, 
                sample_names, 
                grade_names, 
                theme_config, 
                save_prefix
            )
            
        if graph_radar:
            fce.plot_fuzzy_radar(
                B_matrix, 
                sample_names, 
                grade_names, 
                theme_config, 
                save_prefix
            )

    # ---------------------------------------------------------
    # Step 6: Return Full Context
    # ---------------------------------------------------------
    return {
        'Membership_Tensor': R_tensor,    # (n, m, k)
        'Results_Matrix': B_matrix,       # (n, k)
        'Defuzzified_Results': final_results,
        'Raw_Data': data_dict,
        'Config': {
            'grades': grade_names,
            'save_prefix': save_prefix,
            'theme': theme
        }
    }