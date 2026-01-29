import pandas as pd
import os

def process_word_freq_file(file_path):
    """
    Reads an .al file, cleans it, sorts by frequency, and filters for 5-letter words.
    
    Args:
        file_path (str): The path to the .al file.
        
    Returns:
        tuple: (df_full, df_target)
            - df_full: The complete DataFrame (sorted by frequency, re-ranked).
            - df_target: The filtered DataFrame (only 5-letter words, re-ranked).
            Returns (None, None) if an error occurs.
    """
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return None, None

    try:
        # 1. Read the file
        # - sep='\s+': Handles variable whitespace/tabs
        # - usecols=[0, 1, 2]: Discards the 4th column
        # - names: Assigns optimized headers
        df = pd.read_csv(
            file_path, 
            sep='\s+', 
            header=None, 
            usecols=[0, 1, 2], 
            names=['Rank', 'Frequency', 'Word']
        )
        
        # Ensure 'Word' column is string type to avoid length calculation errors
        df['Word'] = df['Word'].astype(str)

        # 2. Global Sort & Re-rank (Step 2 in your logic)
        # Sort by Frequency descending
        df = df.sort_values(by='Frequency', ascending=False)
        
        # Reset index for clean look
        df = df.reset_index(drop=True)
        
        # Re-assign Rank dynamically (1 to N)
        df['Rank'] = range(1, len(df) + 1)
        
        # Save this intermediate state for maintenance
        df_full = df.copy()

        # 3. Filter & Re-sort (Step 3 in your logic)
        # Filter: Keep only words with length == 5
        df_target = df[df['Word'].str.len() == 5].copy()
        
        # Sort again by Frequency descending (safety step)
        df_target = df_target.sort_values(by='Frequency', ascending=False).reset_index(drop=True)
        
        # Re-assign Rank for the new subset (1 to M)
        df_target['Rank'] = range(1, len(df_target) + 1)

        print("Processing completed successfully.")
        return df_full, df_target

    except Exception as e:
        print(f"An error occurred during processing: {e}")
        return None, None

# --- Usage Example ---

# Define your file path
input_file = 'D:/Files/Study/code/DataProcessing/assessment_models/Data_processsing/lemma.al' 

# # Call the function
# # We unpack the result into two variables: 
# # one for maintenance (full_table) and one for your target analysis (len5_table)
# full_table, len5_table = process_word_freq_file(input_file)

# if len5_table is not None:
#     # Preview the Intermediate Table (The full 6000+ list)
#     print("\n--- Intermediate Table (All Words) ---")
#     print(full_table.head())
#     print(f"Total rows: {len(full_table)}")

#     # Preview the Final Target Table (Length = 5)
#     print("\n--- Final Target Table (Length = 5) ---")
#     print(len5_table.head())
#     print(f"Total rows: {len(len5_table)}")

#     len5_table.to_csv('cleared_word_frequency.csv', index=False, encoding='utf-8')
#     print("文件已成功保存为 'cleared_word_frequency.csv'")

# Define your file path
input_file = 'D:/Files/Study/code/DataProcessing/assessment_models/Data_processsing/lemma.al' 

# ---------------------------------------------------------
# 1. 准备工作：确保之前的函数已定义 (或从你的模块导入)
# ---------------------------------------------------------
# 假设 process_word_freq_file 就在同一个脚本里
# 如果在别的文件，请使用: from your_script import process_word_freq_file

# 调用函数获取参照表
# 我们只需要 len5_table (第二个返回值)
_, ref_df = process_word_freq_file(input_file)

if ref_df is None:
    print("错误：无法生成参照词表，请检查 .al 文件路径。")
    exit()

# 为了极速查询，将参照表里的词转为 Set (集合)
# 并统一转为小写 (防止 'Apple' 和 'apple' 对不上)
valid_vocab_set = set(ref_df['Word'].str.lower())

print(f"参照词表加载完毕，包含 {len(valid_vocab_set)} 个有效单词。")

# ---------------------------------------------------------
# 2. 读取 MCM Excel 数据 (你提供的代码)
# ---------------------------------------------------------
excel_path = "D:/Files/Study/code/DataProcessing/assessment_models/Data_processsing/2023_MCM_Problem_C_Data.xlsx"

if not os.path.exists(excel_path):
    print(f"错误：Excel 文件未找到: {excel_path}")
    exit()

df_mcm = pd.read_excel(excel_path, header=1)
df_mcm.columns = df_mcm.columns.map(lambda x: str(x).strip()) # 清理列名空格

# ---------------------------------------------------------
# 3. 核心任务：检验 'Word' 是否在词表内
# ---------------------------------------------------------

# 确保 Excel 里的词也是字符串格式，并统一转为小写进行对比
# 结果存储在 'is_valid' 列：True 表示存在，False 表示不在表里
df_mcm['is_valid'] = df_mcm['Word'].astype(str).str.lower().isin(valid_vocab_set)

# ---------------------------------------------------------
# 4. 查看结果与分析
# ---------------------------------------------------------
print("\n--- 检验完成，前 5 行预览 ---")
print(df_mcm[['Word', 'is_valid']].head())

# 统计一下有多少词不在表里
invalid_count = len(df_mcm[df_mcm['is_valid'] == False])
print(f"\n共有 {invalid_count} 个词未在参考词表中找到。")

if invalid_count > 0:
    print("未找到的词示例：")
    print(df_mcm[df_mcm['is_valid'] == False]['Word'].head().values)

# (可选) 保存结果
# df_mcm.to_excel("Checked_MCM_Data.xlsx", index=False)