import streamlit as st
import numpy as np
import pandas as pd
import sqlite3
import io
from datetime import datetime

# 引入 reportlab 用来动态绘制 PDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# 导入你在 agent_engine.py 中写好的核心算法函数
from agent_engine import extract_live_features, run_statistical_inference, generate_llm_cot

# ==========================================
# 1. 基础设施配置：SQL 数据库与 PDF 字体
# ==========================================
def init_db():
    conn = sqlite3.connect('medical_agent.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            diagnosis TEXT,
            confidence REAL,
            report TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# 注册中文字体（Windows自带微軟正黑體，防止PDF导出时中文变成方块/乱码）
try:
    pdfmetrics.registerFont(TTFont('MSJH', 'C:\\Windows\\Fonts\\msjh.ttc'))
    FONT_NAME = 'MSJH'
except Exception:
    FONT_NAME = 'Helvetica' # 备用标准英文字体

def build_pdf(patient_id, prob, cot_text):
    """将 Agent 思考链文字渲染为正规的 PDF 字节流"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', fontName=FONT_NAME, fontSize=18, leading=22, alignment=1)
    body_style = ParagraphStyle('BodyStyle', fontName=FONT_NAME, fontSize=11, leading=16)
    
    # 写入 PDF 内容头部
    story.append(Paragraph(f"<b>乳腺肿瘤 AI 临床路径推理报告 (CoT)</b>", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"<b>患者病历号 (Patient ID):</b> {patient_id}", body_style))
    story.append(Paragraph(f"<b>报告生成时间:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
    story.append(Paragraph(f"<b>AI 综合评估恶性概率:</b> {prob:.2%}", body_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph(f"<b>【Agent 专家临床思考链意见书】</b>", body_style))
    story.append(Spacer(1, 10))
    
    # 将大模型的换行文本切分并塞进 PDF 段落
    for line in cot_text.split('\n'):
        if line.strip():
            story.append(Paragraph(line, body_style))
            story.append(Spacer(1, 6))
            
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

# ==========================================
# 2. Streamlit 前端交互界面搭建
# ==========================================
st.set_page_config(page_title="乳腺肿瘤 AI 临床推理 Agent 平台", layout="wide")

st.title("🩺 乳腺肿瘤 AI 临床推理 Agent 平台 (V2.0)")
st.markdown("---")

# --- ⚙️ 侧边栏：大模型路由与患者配置面板 ---
st.sidebar.header("🔑 大模型 API 配置")
api_provider = st.sidebar.selectbox(
    "选择 API 供应商", 
    ["DeepSeek (推荐-性价比高)", "阿里云 Qwen", "OpenAI 官方"]
)

# 根据前端选择自动匹配最通用的 Base URL 和 模型简称
if "DeepSeek" in api_provider:
    default_url = "https://api.deepseek.com/v1"
    default_model = "deepseek-chat"
elif "Qwen" in api_provider:
    default_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    default_model = "qwen-max"
else:
    default_url = "https://api.openai.com/v1"
    default_model = "gpt-4o"

user_api_key = st.sidebar.text_input("输入 API Key", type="password", help="在此处粘贴你在大模型官网申请的密钥")
user_base_url = st.sidebar.text_input("Base URL", value=default_url)
user_model_name = st.sidebar.text_input("Model Name", value=default_model)

st.sidebar.markdown("---")
patient_id = st.sidebar.text_input("患者病历号 (Patient ID)", "P_CITYU_2026")

# --- 主界面布局：左右分栏 ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🖼️ 影像数据上传")
    up_img = st.file_uploader("上传病灶局部图 (Crop Image)", type=['png', 'jpg', 'jpeg'])
    up_mask = st.file_uploader("上传病灶掩膜图 (ROI Mask)", type=['png', 'jpg', 'jpeg'])
    
    # 在界面上实时回显图片
    if up_img: st.image(up_img, caption="原始局部病灶 (Crop)", use_container_width=True)
    if up_mask: st.image(up_mask, caption="医生标注掩膜 (Mask)", use_container_width=True)

with col2:
    st.subheader("🧠 Agent 思考链与诊断报告")
    
    if up_img and up_mask:
        if st.button("🚀 启动全链路 Agent 推理", type="primary"):
            if not user_api_key:
                st.warning("⚠️ 提示：您未输入 API Key，系统将使用本地默认模式输出基本统计报告。")
                
            with st.spinner("Agent 正在计算概率特征路径，并同步唤醒专家大模型..."):
                # 1. 读取文件字节流（不落盘）直接由感知层提取几何纹理特征
                img_bytes = up_img.read()
                mask_bytes = up_mask.read()
                feats = extract_live_features(img_bytes, mask_bytes)
                
                if feats:
                    # 2. 统计认知层：计算概率和各变量的回归贡献度
                    prob, contribs, _ = run_statistical_inference(feats)
                    final_decision = "恶性 (MALIGNANT)" if prob > 0.5 else "良性 (BENIGN)"
                    confidence = prob if prob > 0.5 else (1 - prob)
                    
                    # 3. 大模型触达层：通过标准客户端渲染思考链文本
                    cot_report = generate_llm_cot(
                        features=feats, 
                        prob=prob, 
                        contributions=contribs,
                        api_key=user_api_key,
                        base_url=user_base_url,
                        model_name=user_model_name
                    )
                    
                    # 4. 平台前端渲染呈现
                    st.success(f"诊断成功！最终拟诊：**{final_decision}** (置信度: {confidence:.2%})")
                    
                    # 网页直接渲染大模型 CoT 意见书
                    st.markdown("### 📝 临床专家读图意见书")
                    st.info(cot_report)
                    
                    # 5. 实时将大模型生成的报告塞进 PDF 渲染器并生成二进制流
                    pdf_data = build_pdf(patient_id, prob, cot_report)
                    
                    # 6. 一键下载 PDF 报告
                    st.download_button(
                        label="📥 点击下载结构化临床诊断 PDF 意见书",
                        data=pdf_data,
                        file_name=f"Agent_CoT_Report_{patient_id}.pdf",
                        mime="application/pdf"
                    )
                    
                    # 7. SQL 存储层
                    conn = sqlite3.connect('medical_agent.db')
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO records (patient_id, diagnosis, confidence, report, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (patient_id, final_decision, confidence, cot_report, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    conn.commit()
                    conn.close()
                    st.toast("病历数据已通过 SQL 归档至本地数据库！")
                else:
                    st.error("❌ 图像解析失败，请检查上传的二值图掩膜是否包含有效白色轮廓。")
    else:
        st.info("请在左侧侧边栏上传完整的影像和掩膜文件以激活推理 Agent。")

# --- 底部：SQL 历史病历追溯面板 ---
st.markdown("---")
st.subheader("🗄️ 平台历史归档病历 (SQL 实时检索)")
try:
    conn = sqlite3.connect('medical_agent.db')
    history_df = pd.read_sql_query("SELECT id, patient_id, diagnosis, confidence, timestamp FROM records ORDER BY id DESC", conn)
    conn.close()
    st.dataframe(history_df, use_container_width=True)
except Exception:
    st.text("暂无历史数据库记录。")