import pandas as pd

def remove_duplicate_peptides(input_file, output_file, deleted_file):
    """
    删除重复peptide序列，只保留每个序列charge值最小的行
    
    参数:
        input_file: 输入Excel文件名
        output_file: 输出Excel文件名
        deleted_file: 删除记录Excel文件名
    """
    # 读取数据
    df = pd.read_excel(input_file)
    
    # 自动识别peptide和charge列（不区分大小写）
    peptide_col = next((col for col in df.columns if 'peptide' in col.lower()), None)
    charge_col = next((col for col in df.columns if 'charge' in col.lower()), None)
    
    if not peptide_col or not charge_col:
        print("错误: 未找到peptide或charge列")
        return
    
    print(f"使用列名: peptide='{peptide_col}', charge='{charge_col}'")
    print(f"原始数据: {len(df)} 行")
    
    # 找到要保留的行（每个peptide的最小charge）
    keep_idx = df.groupby(peptide_col)[charge_col].idxmin()
    
    # 创建处理后的DataFrame
    df_processed = df.loc[keep_idx].reset_index(drop=True)
    
    # 创建删除记录的DataFrame - 修复：将set转换为list
    deleted_indices = list(set(df.index) - set(keep_idx))
    df_deleted = df.loc[deleted_indices].copy()
    
    # 为删除记录添加原因
    if len(df_deleted) > 0:
        # 获取每个peptide的保留行的charge值
        min_charges = df.groupby(peptide_col)[charge_col].min()
        
        def get_reason(row):
            peptide = row[peptide_col]
            current_charge = row[charge_col]
            min_charge = min_charges[peptide]
            return f"重复序列，charge值较大({current_charge} > {min_charge})"
        
        df_deleted['删除原因'] = df_deleted.apply(get_reason, axis=1)
        
        # 重新排序列，将删除原因放在第一列
        cols = ['删除原因'] + [col for col in df_deleted.columns if col != '删除原因']
        df_deleted = df_deleted[cols]
    
    # 保存文件
    df_processed.to_excel(output_file, index=False)
    print(f"处理后保留: {len(df_processed)} 行")
    
    if len(df_deleted) > 0:
        df_deleted.to_excel(deleted_file, index=False)
        print(f"删除重复: {len(df_deleted)} 行")
        print(f"删除记录已保存到: {deleted_file}")
    else:
        print("没有发现重复序列")
    
    print(f"处理结果已保存到: {output_file}")

# 使用示例
if __name__ == "__main__":
    # 修改这里的文件名
    input_file = "processed_peptides.xlsx"
    output_file = "deduplicated_peptides.xlsx"
    deleted_file = "deleted_duplicates.xlsx"
    
    remove_duplicate_peptides(input_file, output_file, deleted_file)