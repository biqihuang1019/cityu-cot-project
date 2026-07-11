import streamlit as st
import numpy as np
import pandas as pd
import sqlite3
import io
import cv2
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
# 1. 基础设施配置与状态初始化
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
    FONT_NAME = 'Helvetica' 

def build_pdf(patient_id, prob, cot_text):
    """将 Agent 思考链文字渲染为正规的 PDF 字节流"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
    story = []
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('TitleStyle', fontName=FONT_NAME, fontSize=18, leading=22, alignment=1)
    body_style = ParagraphStyle('BodyStyle', fontName=FONT_NAME, fontSize=11, leading=16)
    
    story.append(Paragraph(f"<b>乳腺肿瘤 AI 临床路径推理报告 (CoT)</b>", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"<b>患者病历号 (Patient ID):</b> {patient_id}", body_style))
    story.append(Paragraph(f"<b>报告生成时间:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
    story.append(Paragraph(f"<b>AI 综合评估恶性概率:</b> {prob:.2%}", body_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph(f"<b>【Agent 专家临床思考链意见书】</b>", body_style))
    story.append(Spacer(1, 10))
    
    for line in cot_text.split('\n'):
        if line.strip():
            story.append(Paragraph(line, body_style))
            story.append(Spacer(1, 6))
            
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def safe_load_image(uploaded_file):
    """安全读取上传的文件流，将 16位图像(I;16) 归一化为 8位，并强制用 PNG 编码输出防止前端 JPEG 写入崩溃"""
    # 1. 读取字节流并在不改变原始位深的情况下解码
    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
    
    # 2. 核心修复：如果读取出来是 16位 灰度图，将其映射回 0-255 的 8位图
    if img is not None and (img.dtype == np.uint16 or (len(img.shape) == 2 and img.itemsize == 2)):
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    elif img is not None and img.dtype != np.uint8:
        img = img.astype(np.uint8)
        
    # 3. 极其重要：重置文件指针，确保后续的业务推理代码（如 agent 提取特征）依然能够读取文件
    uploaded_file.seek(0)
    
    # 4. 转成对高位深兼容的 PNG 字节流供给 st.image 渲染
    if img is not None:
        _, encoded_img = cv2.imencode('.png', img)
        return encoded_img.tobytes()
    return None

# --- 初始化 Session State（防止点击下载或浏览器翻译时组件消失触发 removeChild 崩溃） ---
if "agent_results" not in st.session_state:
    st.session_state.agent_results = None

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
    ["DeepSeek (推荐-性价比高)", "阿里云 Qwen", "OpenAI 官方"],
    key="api_provider_select"
)

if "DeepSeek" in api_provider:
    default_url = "https://api.deepseek.com/v1"
    default_model = "deepseek-chat"
elif "Qwen" in api_provider:
    default_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    default_model = "qwen-max"
else:
    default_url = "https://api.openai.com/v1"
    default_model = "gpt-4o"

user_api_key = st.sidebar.text_input("输入 API Key", type="password", help="在此处粘贴你在大模型官网申请的密钥", key="api_key_input")
user_base_url = st.sidebar.text_input("Base URL", value=default_url, key="base_url_input")
user_model_name = st.sidebar.text_input("Model Name", value=default_model, key="model_name_input")

st.sidebar.markdown("---")
patient_id = st.sidebar.text_input("患者病历号 (Patient ID)", "P_CITYU_2026", key="patient_id_input")

# --- 主界面布局：左右分栏 ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("🖼️ 影像数据上传")
    up_img = st.file_uploader("上传病灶局部图 (Crop Image)", type=['png', 'jpg', 'jpeg'], key="upload_img_file")
    up_mask = st.file_uploader("上传病灶掩膜图 (ROI Mask)", type=['png', 'jpg', 'jpeg'], key="upload_mask_file")
    
    # 采用安全预览机制包裹图像渲染
    if up_img:
        try:
            safe_img_bytes = safe_load_image(up_img)
            if safe_img_bytes:
                st.image(safe_img_bytes, caption="原始局部病灶 (Crop)", use_container_width=True)
        except Exception as e:
            st.error(f"图像预览失败: {e}")
            
    if up_mask:
        try:
            safe_mask_bytes = safe_load_image(up_mask)
            if safe_mask_bytes:
                st.image(safe_mask_bytes, caption="医生标注掩膜 (Mask)", use_container_width=True)
        except Exception as e:
            st.error(f"掩膜预览失败: {e}")

with col2:
    st.subheader("🧠 Agent 思考链与诊断报告")
    
    if up_img and up_mask:
        # 点击按钮执行计算，并固化结果到状态机
        if st.button("🚀 启动全链路 Agent 推理", type="primary", key="trigger_agent_btn"):
            if not user_api_key:
                st.warning("⚠️ 提示：您未输入 API Key，系统将使用本地默认模式输出基本统计报告。")
                
            with st.spinner("Agent 正在计算概率特征路径，并同步唤醒专家大模型..."):
                img_bytes = up_img.read()
                mask_bytes = up_mask.read()
                
                # 读取后重置，防止按钮重新触发时为空
                up_img.seek(0)
                up_mask.seek(0)
                
                feats = extract_live_features(img_bytes, mask_bytes)
                
                if feats:
                    prob, contribs, _ = run_statistical_inference(feats)
                    final_decision = "恶性 (MALIGNANT)" if prob > 0.5 else "良性 (BENIGN)"
                    confidence = prob if prob > 0.5 else (1 - prob)
                    
                    cot_report = generate_llm_cot(
                        features=feats, 
                        prob=prob, 
                        contributions=contribs,
                        api_key=user_api_key,
                        base_url=user_base_url,
                        model_name=user_model_name
                    )
                    
                    pdf_data = build_pdf(patient_id, prob, cot_report)
                    
                    # 写入历史数据库
                    conn = sqlite3.connect('medical_agent.db')
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO records (patient_id, diagnosis, confidence, report, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (patient_id, final_decision, confidence, cot_report, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    conn.commit()
                    conn.close()
                    
                    # 固化状态
                    st.session_state.agent_results = {
                        "final_decision": final_decision,
                        "confidence": confidence,
                        "cot_report": cot_report,
                        "pdf_data": pdf_data,
                        "patient_id": patient_id
                    }
                    st.toast("病历数据已通过 SQL 归档至本地数据库！")
                else:
                    st.error("❌ 图像解析失败，请检查上传的二值图掩膜是否包含有效白色轮廓。")
        
        # 结果渲染区（脱离了 st.button 的if生命周期，完美规避页面刷新导致的闪退崩溃）
        if st.session_state.agent_results is not None:
            res = st.session_state.agent_results
            st.success(f"诊断成功！最终拟诊：**{res['final_decision']}** (置信度: {res['confidence']:.2%})")
            
            st.markdown("### 📝 临床专家读图意见书")
            st.info(res['cot_report'], icon="📄")
            
            st.download_button(
                label="📥 点击下载结构化临床诊断 PDF 意见书",
                data=res['pdf_data'],
                file_name=f"Agent_CoT_Report_{res['patient_id']}.pdf",
                mime="application/pdf",
                key="download_pdf_report_btn"
            )
    else:
        # 上传文件缺失时清除上一次的遗留状态
        st.session_state.agent_results = None
        st.info("请在左侧侧边栏上传完整的影像和掩膜文件以激活推理 Agent。")

# --- 底部：SQL 历史病历追溯面板 ---
st.markdown("---")
st.subheader("🗄️ 平台历史归档病历 (SQL 实时检索)")
try:
    conn = sqlite3.connect('medical_agent.db')
    history_df = pd.read_sql_query("SELECT id, patient_id, diagnosis, confidence, timestamp FROM records ORDER BY id DESC", conn)
    conn.close()
    # 补全关键 key，防止表格刷新与 DOM 碰撞
    st.dataframe(history_df, use_container_width=True, key="sql_history_table")
except Exception:
    st.text("暂无历史数据库记录。")