import cv2
import numpy as np
import pandas as pd
import os
from skimage.feature import graycomatrix, graycoprops
from openai import OpenAI

# 假设这是你用上万例数据训练出来的逻辑回归系数（标准化后）
BETA_DICT = {
    'Circularity': 1.45,   # 圆形度
    'Solidity': -1.12,    # 凸包度
    'GLCM_Contrast': 0.85 # 对比度
}

def extract_live_features(img_bytes, mask_bytes):
    """实时感知层：从用户上传的字节流中提取数学特征"""
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    
    if img is None or mask is None:
        return None
        
    _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    c = max(contours, key=cv2.contourArea)
    
    # 几何计算
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    hull = cv2.convexHull(c)
    solidity = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
    
    # 纹理计算
    img_coarse = (img // 4).astype(np.uint8)
    glcm = graycomatrix(img_coarse, distances=[1], angles=[0], levels=64, symmetric=True, normed=True)
    contrast = np.mean(graycoprops(glcm, 'contrast'))
    
    return {'Circularity': circularity, 'Solidity': solidity, 'GLCM_Contrast': contrast}

def run_statistical_inference(features):
    """认知推理层：计算条件概率与各指标贡献度"""
    z_scores = {
        'Circularity': (features['Circularity'] - 0.55) / 0.15,
        'Solidity': (features['Solidity'] - 0.88) / 0.08,
        'GLCM_Contrast': (features['GLCM_Contrast'] - 2.5) / 1.1
    }
    
    log_odds = 0.1 # 截距
    contributions = {}
    for k, v in z_scores.items():
        contrib = BETA_DICT[k] * v
        contributions[k] = contrib
        log_odds += contrib
        
    prob_malignant = 1 / (1 + np.exp(-log_odds))
    return prob_malignant, contributions, z_scores

def generate_llm_cot(features, prob, contributions, api_key=None, base_url=None, model_name="gpt-4o"):
    """认知触达层：大模型生成医学级思考链报告"""
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", "your-default-key-here")
    if not base_url:
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

    client = OpenAI(api_key=api_key, base_url=base_url)
    
    prompt = f"""
你是一位世界顶尖的乳腺放射科专家。请根据以下由计算机视觉算法（OpenCV）提取的肿瘤数学特征，以及逻辑回归模型计算出的统计学贡献度，为医生撰写一份结构化的“读图思考链 (CoT) 临床意见书”。

【底层感知数学矩阵】
- 肿瘤圆形度 (Circularity): {features['Circularity']:.2f} 
- 凸包凹陷度 (Solidity): {features['Solidity']:.2f}
- 纹理灰度对比度 (GLCM_Contrast): {features['GLCM_Contrast']:.2f}

【认知模型因果推理数据】
- 模型综合预测恶性概率 (Prob_Malignant): {prob:.2%}
- 特征决策贡献度 (Contributions): {contributions} 
  *(注：贡献度正值越大代表该指标越支持恶性判定，负值绝对值越大越支持良性判定。)*

【医学报告撰写规范要求】
1. 【图像病灶形态学分析】：请用精炼的医学黑话解释上述形态学指标。例如：Circularity 偏低在临床上对应“边缘欠规则、呈蟹足状或毛刺状浸润生长”；Solidity 降低对应“边界不规则凹陷”；GLCM_Contrast 对应“内部灰度异质性与微钙化风险”。
2. 【鉴别诊断决策博弈】：展现你（Agent）是如何在支持恶性的指标（正贡献度）与支持良性的指标（负贡献度）之间进行统计学权衡和逻辑博弈的。
3. 【最终拟诊意见】：基于上述推理路径，给出一个确切的 BI-RADS 分级倾向（如 BI-RADS 4 或 BI-RADS 2），并给出临床后续处置建议（如 3个月随访 或 安排活检）。
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一个高度严谨、只基于硬核统计学数据和临床医学事实进行推理的医疗大模型 Agent。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1200
        )
        return response.choices[0].message.content
        
    except Exception as e:
        return f"【⚠️ API 接入异常提示】由于网络或鉴权原因未能成功唤醒大模型，以下为您提供纯统计学推理路径：\n\n" \
               f"该病例的最终预测结论受 [Circularity] 贡献度({contributions.get('Circularity', 0):+.2f}) 与 " \
               f"[Solidity] 贡献度({contributions.get('Solidity', 0):+.2f}) 联合博弈决定。最终收敛恶性概率为 {prob:.2%}。"