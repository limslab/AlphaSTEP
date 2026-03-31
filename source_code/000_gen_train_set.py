import pandas as pd

# 读取第一个Excel文件的第一列数据(不包括表头)
df1 = pd.read_excel('非.xlsx', usecols=[0], header=None, skiprows=1)
X = df1[0].unique()

# 读取第二个Excel文件的第一列数据(不包括表头)
df2 = pd.read_excel('combined_data.xlsx', usecols=[0], header=None, skiprows=1)
X2 = df2[0].unique()

# 创建一个字典,用于存储X和对应的y值
# 调整：第一个文件(X)中的值标记为0，第二个文件(X2)中的值标记为1
all_elements = set(X).union(set(X2))
data = {x: 1 if x in X2 else 0 for x in all_elements}  # 这里逻辑保持不变，X2中的值为1，X中独有的为0

# 将字典转换为DataFrame
result = pd.DataFrame.from_dict(data, orient='index', columns=['y'])
result.index.name = 'X'

# 重置索引,使X成为一个普通的列
result.reset_index(inplace=True)

# 保存结果为CSV文件
result.to_csv('result.csv', index=False)
