import pandas as pd

def process_and_deduplicate_peptides(input_file, output_file, deleted_file):
    """
    1. 过滤掉包含多个蛋白（以分号分隔）的行
    2. 删除重复peptide序列，只保留每个序列charge值最小的行
    
    参数:
        input_file: 输入Excel文件名
        output_file: 输出Excel文件名
        deleted_file: 删除记录Excel文件名
    """
    # 读取数据
    print(f"正在读取文件: {input_file}")
    df = pd.read_excel(input_file)
    
    # ================== 列名重命名与格式清理 ==================
    # 1. 识别现有的 sequence 列，并将其列名改为 'Peptide'
    seq_col = next((col for col in df.columns if col.lower() == 'sequence'), None)
    if seq_col:
        df.rename(columns={seq_col: 'Peptide'}, inplace=True)
    
    # 2. 清理 Peptide 序列格式，将 b'ICTDIGK' 转换为 ICTDIGK
    if 'Peptide' in df.columns:
        df['Peptide'] = df['Peptide'].astype(str).str.replace(r"^b['\"](.*)['\"]$", r"\1", regex=True)

    # 3. 自动识别peptide, charge, protein列（不区分大小写）
    peptide_col = next((col for col in df.columns if 'peptide' in col.lower()), None)
    charge_col = next((col for col in df.columns if 'charge' in col.lower()), None)
    protein_col = next((col for col in df.columns if 'protein' in col.lower()), None)
    
    if not peptide_col or not charge_col:
        print("错误: 未找到peptide或charge列")
        return
    
    print(f"使用列名: peptide='{peptide_col}', charge='{charge_col}', protein='{protein_col}'")
    print(f"原始数据: {len(df)} 行")

    # 准备一个列表，用于收集所有被删除的DataFrame片段
    deleted_frames = []

    # ================== 步骤一：过滤多蛋白 ==================
    if protein_col:
        # 顺手清理一下 protein 列的 b'' 格式，防止干扰分号的识别
        df[protein_col] = df[protein_col].astype(str).str.replace(r"^b['\"](.*)['\"]$", r"\1", regex=True)
        
        # 查找包含分号的行（多蛋白）
        multi_protein_mask = df[protein_col].str.contains(';', na=False)
        
        # 将多蛋白行记录到删除列表
        df_multi_protein = df[multi_protein_mask].copy()
        if not df_multi_protein.empty:
            df_multi_protein['删除原因'] = '包含多个蛋白'
            deleted_frames.append(df_multi_protein)
            print(f"发现并过滤多蛋白数据: {len(df_multi_protein)} 行")
        
        # 剔除多蛋白数据，更新df，准备进行下一步去重
        df = df[~multi_protein_mask].copy()
    else:
        print("警告: 未找到protein列，跳过多蛋白过滤步骤。")

    # ================== 步骤二：去重（保留最小charge） ==================
    print(f"参与去重的单蛋白数据: {len(df)} 行")
    
    if len(df) > 0:
        # 找到要保留的行（每个peptide的最小charge）
        keep_idx = df.groupby(peptide_col)[charge_col].idxmin()
        
        # 创建处理后的最终DataFrame
        df_processed = df.loc[keep_idx].reset_index(drop=True)
        
        # 获取因为重复被删除的行
        deleted_indices_dup = list(set(df.index) - set(keep_idx))
        df_deleted_dup = df.loc[deleted_indices_dup].copy()
        
        # 为重复删除记录添加原因
        if len(df_deleted_dup) > 0:
            min_charges = df.groupby(peptide_col)[charge_col].min()
            
            def get_reason(row):
                peptide = row[peptide_col]
                current_charge = row[charge_col]
                min_charge = min_charges[peptide]
                return f"重复序列，charge值较大({current_charge} > {min_charge})"
            
            df_deleted_dup['删除原因'] = df_deleted_dup.apply(get_reason, axis=1)
            deleted_frames.append(df_deleted_dup)
            print(f"发现并过滤重复序列数据: {len(df_deleted_dup)} 行")
    else:
        # 如果过滤完多蛋白后没数据了
        df_processed = df.copy()

    # ================== 步骤三：保存结果 ==================
    # 处理并保存删除记录
    if deleted_frames:
        # 将多蛋白删除记录和重复删除记录合并成一个表
        df_all_deleted = pd.concat(deleted_frames, ignore_index=True)
        
        # 重新排序列，将'删除原因'移动到第一列方便查看
        cols = ['删除原因'] + [col for col in df_all_deleted.columns if col != '删除原因']
        df_all_deleted = df_all_deleted[cols]
        
        df_all_deleted.to_excel(deleted_file, index=False)
        print(f"总计删除: {len(df_all_deleted)} 行，删除记录已保存到: {deleted_file}")
    else:
        print("太棒了，没有发现需要删除的异常或重复数据！")

    # 保存处理后的文件
    df_processed.to_excel(output_file, index=False)
    print(f"处理后最终保留: {len(df_processed)} 行")
    print(f"最终结果已保存到: {output_file}")


# 使用示例
if __name__ == "__main__":
    # 修改这里的文件名
    input_file = "precursor_df_output1.xlsx"
    output_file = "filtered_and_deduplicated.xlsx"
    deleted_file = "deleted_records.xlsx"
    
    process_and_deduplicate_peptides(input_file, output_file, deleted_file)