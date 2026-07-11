import os
import pandas as pd
import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm

# ==========================================
# 1. 路径配置 (根据你的电脑实际环境)
# ==========================================
BASE_DIR = r"D:\biqi\cityU\study\summer term\COT\cbis-ddsm-1024fixed2\versions\1\CBIS-DDSM-1024fixed2"
INDEX_CSV = os.path.join(BASE_DIR, "CBIS_Master_Index.csv")

# 上一轮写好的特征提取核心函数
def extract_features(roi_local_path, crop_local_path):
    try:
        # 读取图像
        mask = cv2.imread(roi_local_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(crop_local_path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None or img is None:
            return None
            
        # A. 几何特征 (基于ROI二值图)
        _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        hull = cv2.convexHull(c)
        solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
        
        # B. 纹理特征 (基于Crop局部灰度图)
        img_coarse = (img // 4).astype(np.uint8) # 压缩灰度级
        glcm = graycomatrix(img_coarse, distances=[1], angles=[0, np.pi/2], levels=64, symmetric=True, normed=True)
        contrast = np.mean(graycoprops(glcm, 'contrast'))
        energy = np.mean(graycoprops(glcm, 'energy'))
        
        return [area, perimeter, circularity, solidity, contrast, energy]
    except Exception as e:
        return None

# ==========================================
# 2. 读取总表并修正路径
# ==========================================
print("正在读取索引总表...")
# 如果你的文件是Excel，用 pd.read_excel(INDEX_CSV)；如果是CSV，用 pd.read_csv
df = pd.read_csv(INDEX_CSV) if INDEX_CSV.endswith('.csv') else pd.read_excel(INDEX_CSV)

# 转换标签为数学信号：MALIGNANT -> 1 (恶性), BENIGN 等 -> 0 (良性)
df['Label'] = df['Pathology'].apply(lambda x: 1 if str(x).upper() == 'MALIGNANT' else 0)

features_list = []

print("开始批量跨文件夹提取图像数学特征...")
# 使用 tqdm 显示进度条
for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    # 核心：将原作者的 C 盘绝对路径转换为你本地的 D 盘相对/绝对路径
    # 假设原作者路径里含有 'roi' 或 'cropped' 文件夹名
    roi_sub_path = row['ROI_Path'].split('CBIS-DDSM-1024fixed2')[-1].lstrip('\\/')
    crop_sub_path = row['Crop_Path'].split('CBIS-DDSM-1024fixed2')[-1].lstrip('\\/')
    
    local_roi = os.path.join(BASE_DIR, roi_sub_path)
    local_crop = os.path.join(BASE_DIR, crop_sub_path)
    
    # 执行提取
    feats = extract_features(local_roi, local_crop)
    
    if feats is not None:
        # 将表格原有的病理学诊断指标与提取出的数学特征合并
        full_row = [
            row['PatientID'], row['Assessment'], row['Label'], row['Split']
        ] + feats
        features_list.append(full_row)

# ==========================================
# 3. 导出全新的统计分析矩阵
# ==========================================
col_names = ['PatientID', 'Assessment', 'Label', 'Split', 
             'Area', 'Perimeter', 'Circularity', 'Solidity', 'GLCM_Contrast', 'GLCM_Energy']

result_df = pd.DataFrame(features_list, columns=col_names)
output_path = os.path.join(BASE_DIR, "Breast_Cancer_Mathematical_Matrix.csv")
result_df.to_csv(output_path, index=False)

print(f"\n【大功告成】数学矩阵已生成！共成功处理 {len(result_df)} 例样本。")
print(f"数据已保存在: {output_path}")