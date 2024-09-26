import os
import shutil

# 指定你想要操作的文件夹路径
folder_path = '/home/zhushenghao/data/JS/Brats2024/training_data1_v2'

# 遍历指定文件夹下的所有子文件夹
for subdir, dirs, files in os.walk(folder_path):
    # 跳过已经是 "nii" 的文件夹
    if os.path.basename(subdir) == 'nii':
        continue

    # 创建 "nii" 文件夹的路径
    nii_folder_path = os.path.join(subdir, 'nii')

    # 如果 "nii" 文件夹不存在，则创建它
    if not os.path.exists(nii_folder_path):
        os.makedirs(nii_folder_path)

    # 遍历子文件夹中的所有文件
    for item in files:
        # 构建文件的完整路径
        item_path = os.path.join(subdir, item)

        # 检查它是否是一个文件
        if os.path.isfile(item_path):
            # 移动文件到 "nii" 文件夹
            shutil.move(item_path, nii_folder_path)
            print(f"文件 {item} 已移动到 {nii_folder_path}")

if os.path.exists(os.path.join(folder_path, 'nii')):
    # 删除 "nii" 文件夹
    shutil.rmtree(os.path.join(folder_path, 'nii'))

print("所有文件已成功移动到各自子文件夹下的 'nii' 文件夹。")

