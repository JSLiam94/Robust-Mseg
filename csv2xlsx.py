import pandas as pd


def csv_to_xlsx(csv_file_path, xlsx_file_path):
    # 读取CSV文件
    df = pd.read_csv(csv_file_path, delimiter=';')

    # 将DataFrame保存为Excel文件
    df.to_excel(xlsx_file_path, index=False)


# 使用示例
csv_file_path = '/home/zhushenghao/data/JS/Robust-Mseg-main/output/20240923-131430/test_pred/results.csv'  # 替换为你的CSV文件路径
xlsx_file_path = '/home/zhushenghao/data/JS/Robust-Mseg-main/output/20240923-131430/test_pred/results.xlsx'  # 替换为你想要保存的Excel文件路径

csv_to_xlsx(csv_file_path, xlsx_file_path)
