import os

os.environ["KAGGLEHUB_CACHE_DIR"] = r"D:\biqi\cityU\study\summer term\COT\kaggle_cache"

import kagglehub

print("【提示】正在连接 Kaggle 服务器并下载数据集，请保持网络畅通...")
print("【目标路径】:", os.environ["KAGGLEHUB_CACHE_DIR"])

path = kagglehub.dataset_download("abdelrahmanelmugh/cbis-ddsm-1024fixed2")

print("\n" + "="*50)
print("【成功】数据集下载及解压完成！")
print("【数据集实际存放绝对路径】:", path)
print("="*50)