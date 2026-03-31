import h5py
import pandas as pd

# 替换为你的HDF文件路径
hdf_file_path = 'output_spectral_library2.hdf'

# 用于存储所有数据集数据的字典
data_dict = {}

# 打开HDF文件
with h5py.File(hdf_file_path, 'r') as hf:
    precursor_df_group = hf['library/precursor_df']
    for dataset_name in precursor_df_group.keys():
        data_dict[dataset_name] = precursor_df_group[dataset_name][:]

# 将字典转换为DataFrame
precursor_df = pd.DataFrame(data_dict)

# 定义要保存的Excel文件路径
excel_file_path = 'precursor_df_output.xlsx'

# Excel 单个工作表的最大行数
max_rows = 1048500

# 计算需要的工作表数量
num_sheets = len(precursor_df) // max_rows + (1 if len(precursor_df) % max_rows != 0 else 0)

# 使用 ExcelWriter 写入多个工作表
with pd.ExcelWriter(excel_file_path) as writer:
    for i in range(num_sheets):
        start_row = i * max_rows
        end_row = min((i + 1) * max_rows, len(precursor_df))
        subset_df = precursor_df.iloc[start_row:end_row]
        sheet_name = f'Sheet{i + 1}'
        # 指定起始行，表头从第 1 行开始，数据从第 2 行开始
        subset_df.to_excel(writer, sheet_name=sheet_name, index=False, startrow=0)

print(f"数据已成功保存到 {excel_file_path}")