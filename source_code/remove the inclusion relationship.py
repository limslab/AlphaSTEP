import pandas as pd
import os

def filter_peptide_sequences_with_deleted_records(excel_file_path):
    """
    筛选肽段序列：如果长肽段包含了短肽段，则删除长肽段，保留短肽段
    返回：筛选后的DataFrame和删除记录的DataFrame
    """
    # 读取 Excel 文件
    df = pd.read_excel(excel_file_path)
    
    # 提取肽段序列及其原始索引
    peptide_pairs = list(zip(df['Peptide'].tolist(), df.index.tolist()))

    # 对肽段序列按长度从长到短排序
    peptide_pairs.sort(key=lambda x: len(x[0]), reverse=True)

    # 初始化结果和删除记录
    result_indices = []
    deleted_records = []
    
    # 创建一个集合来跟踪已处理的索引
    processed_indices = set()
    
    # 从最长到最短处理肽段
    for i in range(len(peptide_pairs)):
        current_peptide, current_idx = peptide_pairs[i]
        
        # 如果当前索引已处理过，跳过
        if current_idx in processed_indices:
            continue
        
        # 标记当前索引为已处理
        processed_indices.add(current_idx)
        
        # 查找所有比当前肽段短的包含关系
        shorter_contained = False
        contained_by_indices = []
        contained_by_peptides = []
        
        for j in range(i + 1, len(peptide_pairs)):
            shorter_peptide, shorter_idx = peptide_pairs[j]
            
            # 如果短肽段已被处理，跳过
            if shorter_idx in processed_indices:
                continue
            
            # 检查包含关系：当前肽段是否包含短肽段
            if shorter_peptide in current_peptide:
                shorter_contained = True
                contained_by_indices.append(shorter_idx)
                contained_by_peptides.append(shorter_peptide)
                processed_indices.add(shorter_idx)
        
        if shorter_contained:
            # 当前肽段包含了更短的肽段，删除当前肽段，保留短肽段
            deleted_record = {
                'deleted_index': current_idx,
                'deleted_peptide': current_peptide,
                'deleted_reason': '被保留的较短肽段包含',
                'related_indices': str(contained_by_indices),
                'related_peptides': str(contained_by_peptides),
                'relationship': f"'{current_peptide}' 包含以下较短肽段"
            }
            deleted_records.append(deleted_record)
            
            # 保留所有较短的肽段
            result_indices.extend(contained_by_indices)
        else:
            # 没有包含更短的肽段，保留当前肽段
            result_indices.append(current_idx)
    
    # 根据筛选后的索引获取对应的行
    result_df = df.loc[sorted(result_indices)]
    
    # 创建删除记录DataFrame
    if deleted_records:
        deleted_df = pd.DataFrame(deleted_records)
        
        # 获取所有被删除行的完整信息
        deleted_indices = deleted_df['deleted_index'].tolist()
        deleted_rows = df.loc[deleted_indices].copy()
        
        # 合并原始数据和删除原因
        deleted_rows = deleted_rows.reset_index(drop=True)
        deleted_df = deleted_df.reset_index(drop=True)
        
        deleted_df = pd.concat([deleted_df[['deleted_index', 'deleted_reason', 
                                          'related_indices', 'related_peptides', 
                                          'relationship']], 
                               deleted_rows], axis=1)
        
        deleted_df = deleted_df.rename(columns={
            'deleted_index': '原始行索引',
            'deleted_reason': '删除原因',
            'related_indices': '相关保留行索引',
            'related_peptides': '相关保留肽段序列',
            'relationship': '包含关系描述'
        })
    else:
        deleted_df = pd.DataFrame(columns=['原始行索引', '删除原因', '相关保留行索引', 
                                          '相关保留肽段序列', '包含关系描述'])
    
    return result_df, deleted_df


def iterative_filter_peptide_sequences(excel_file_path):
    """
    迭代处理包含关系，直到没有新的删除发生
    """
    current_file = excel_file_path
    all_deleted_records = []
    iteration = 1
    
    while True:
        print(f"\n第 {iteration} 次迭代处理...")
        
        # 应用筛选
        result_df, deleted_df = filter_peptide_sequences_with_deleted_records(current_file)
        
        if deleted_df.empty:
            print("没有发现新的包含关系，处理完成。")
            break
        
        print(f"本轮删除了 {len(deleted_df)} 行")
        
        # 保存本轮结果到临时文件
        temp_output = f'temp_iteration_{iteration}.xlsx'
        result_df.to_excel(temp_output, index=False)
        
        # 保存本轮删除记录
        temp_deleted = f'temp_deleted_iteration_{iteration}.xlsx'
        deleted_df.to_excel(temp_deleted, index=False)
        
        # 记录删除信息
        all_deleted_records.append((iteration, deleted_df))
        
        # 为下一轮准备
        current_file = temp_output
        iteration += 1
    
    # 合并所有删除记录
    if all_deleted_records:
        final_deleted_df = pd.concat([df for _, df in all_deleted_records], ignore_index=True)
    else:
        final_deleted_df = pd.DataFrame()
    
    return result_df, final_deleted_df, iteration-1  # 返回迭代次数


def cleanup_temp_files(max_iterations):
    """
    清理临时文件
    """
    for i in range(1, max_iterations + 1):
        temp_file = f'temp_iteration_{i}.xlsx'
        temp_deleted = f'temp_deleted_iteration_{i}.xlsx'
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"已删除临时文件: {temp_file}")
        if os.path.exists(temp_deleted):
            os.remove(temp_deleted)
            print(f"已删除临时文件: {temp_deleted}")


# 主程序
if __name__ == "__main__":
    # 输入文件路径
    excel_file_path = 'filtered_and_deduplicated.xlsx'
    
    # 检查文件是否存在
    if not os.path.exists(excel_file_path):
        print(f"错误: 文件 {excel_file_path} 不存在!")
        exit(1)
    
    print(f"开始处理文件: {excel_file_path}")
    print("=" * 50)
    
    # 使用迭代处理
    filtered_df, deleted_df, total_iterations = iterative_filter_peptide_sequences(excel_file_path)
    
    # 保存最终结果
    output_file_path = 'unqiue peptide.xlsx'
    filtered_df.to_excel(output_file_path, index=False)
    
    # 保存所有删除记录
    deleted_records_path = 'delete record.xlsx'
    if not deleted_df.empty:
        deleted_df.to_excel(deleted_records_path, index=False)
        print(f"\n所有删除记录已保存到 {deleted_records_path}")
    else:
        print("\n没有删除任何行")
    
    print(f"最终结果已保存到 {output_file_path}")
    print(f"总迭代次数: {total_iterations}")
    print(f"筛选后保留 {len(filtered_df)} 行，总共删除了 {len(deleted_df)} 行")
    
    # 清理临时文件
    print("\n清理临时文件...")
    cleanup_temp_files(total_iterations)
    
    print("\n处理完成!")
    print("=" * 50)