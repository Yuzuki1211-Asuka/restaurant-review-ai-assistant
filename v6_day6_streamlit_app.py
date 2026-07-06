import os
import json
import ast
from pathlib import Path

import pandas as pd
import streamlit as st

# 可选依赖：如果环境里没有 joblib / jieba，也不会影响展示型页面
try:
    import joblib
except Exception:
    joblib = None

try:
    import jieba
except Exception:
    jieba = None


# =========================
# 基础配置
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
MODELS_DIR = OUTPUTS_DIR / "models"
RAG_DIR = OUTPUTS_DIR / "rag_results"

st.set_page_config(
    page_title="校园周边餐厅智能评价分析助手",
    page_icon="🍜",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# 通用工具函数
# =========================
def file_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def show_image(path: Path, caption: str = ""):
    if file_exists(path):
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"缺少图片：{path}")


def read_json(path: Path):
    if not file_exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_data():
    path = DATA_DIR / "processed_reviews.csv"
    if not file_exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_data
def load_csv(path_str: str):
    path = Path(path_str)
    if not file_exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


@st.cache_resource
def load_ml_assets():
    """
    尝试加载 v3 保存的 TF-IDF 和最优模型。
    如果当前环境和训练环境 sklearn 版本不一致，可能加载失败。
    展示型 v6 允许失败，不影响其他页面。
    """
    if joblib is None:
        return None, None, "当前环境没有 joblib，无法加载 ML 模型。"

    tfidf_path = MODELS_DIR / "tfidf_vectorizer.pkl"
    model_path = MODELS_DIR / "best_model.pkl"

    if not file_exists(tfidf_path) or not file_exists(model_path):
        return None, None, "缺少 tfidf_vectorizer.pkl 或 best_model.pkl。"

    try:
        tfidf = joblib.load(tfidf_path)
        model = joblib.load(model_path)
        return tfidf, model, ""
    except Exception as e:
        return None, None, f"模型加载失败：{e}"


def simple_tokenize(text: str) -> str:
    """尽量使用 jieba 分词，以匹配 v3 的 TF-IDF 输入格式。"""
    text = str(text).strip()
    if not text:
        return ""

    userdict = DATA_DIR / "userdict_restaurant.txt"
    if jieba is not None:
        try:
            if userdict.exists():
                jieba.load_userdict(str(userdict))
            return " ".join([w for w in jieba.lcut(text) if w.strip()])
        except Exception:
            return text
    return text


def sentiment_label(pred):
    try:
        pred = int(pred)
    except Exception:
        pass

    if pred == 1 or pred == "positive":
        return "正面评论"
    if pred == 0 or pred == "negative":
        return "负面评论"
    return str(pred)


# =========================
# 加载数据
# =========================
df = load_data()


# =========================
# 侧边栏
# =========================
st.sidebar.title("🍜 餐厅评价分析助手")
st.sidebar.caption("v6 展示型系统集成版")

page = st.sidebar.radio(
    "功能导航",
    [
        "🏠 首页概览",
        "📊 数据看板",
        "🖼️ EDA 图表",
        "🤖 ML 情感预测",
        "🧠 DL 深度学习分析",
        "💬 RAG 问答结果",
        "📈 融合对比",
        "📁 产物检查"
    ]
)

st.sidebar.divider()
st.sidebar.write("当前项目目录：")
st.sidebar.code(str(BASE_DIR))


# =========================
# 首页
# =========================
if page == "🏠 首页概览":
    st.title("校园周边餐厅智能评价分析助手")
    st.markdown(
        """
        本系统集成了前 5 个阶段的实训成果，形成一个可交互的 Streamlit 展示应用。

        **系统包含：**

        - v1：数据预处理结果展示  
        - v2：EDA 可视化图表展示  
        - v3：机器学习情感分类结果与可选预测  
        - v4：深度学习训练曲线与注意力可视化  
        - v5：RAG 问答结果与证据链展示  
        """
    )

    if df.empty:
        st.error("未找到 data/processed_reviews.csv，请检查路径。")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("评论总数", f"{len(df):,}")
        c2.metric("餐厅数量", df["restaurant_name"].nunique() if "restaurant_name" in df.columns else "-")
        c3.metric("菜系数量", df["cuisine_type"].nunique() if "cuisine_type" in df.columns else "-")
        c4.metric("平均评分", f"{df['rating'].mean():.2f}" if "rating" in df.columns else "-")

    st.subheader("项目流程")
    st.markdown(
        """
        `数据清洗 → EDA 可视化 → 机器学习分类 → 深度学习分类 → RAG 问答 → Streamlit 系统集成`
        """
    )


# =========================
# 数据看板
# =========================
elif page == "📊 数据看板":
    st.title("📊 数据看板")

    if df.empty:
        st.error("未找到 processed_reviews.csv。")
        st.stop()

    if "rating" in df.columns:
        avg_rating = df["rating"].mean()
        positive_rate = (df["rating"] >= 4).mean()
        negative_rate = (df["rating"] <= 2).mean()
    else:
        avg_rating = 0
        positive_rate = 0
        negative_rate = 0

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("评论总数", f"{len(df):,}")
    c2.metric("餐厅数", df["restaurant_name"].nunique() if "restaurant_name" in df.columns else "-")
    c3.metric("菜系数", df["cuisine_type"].nunique() if "cuisine_type" in df.columns else "-")
    c4.metric("平均评分", f"{avg_rating:.2f}")
    c5.metric("好评率", f"{positive_rate:.1%}")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("评分分布")
        if "rating" in df.columns:
            rating_counts = df["rating"].value_counts().sort_index()
            st.bar_chart(rating_counts)
        else:
            st.warning("缺少 rating 字段。")

    with col2:
        st.subheader("菜系评论量 Top10")
        if "cuisine_type" in df.columns:
            cuisine_counts = df["cuisine_type"].value_counts().head(10)
            st.bar_chart(cuisine_counts)
        else:
            st.warning("缺少 cuisine_type 字段。")

    st.subheader("数据预览")
    st.dataframe(df.head(50), use_container_width=True)


# =========================
# EDA 图表
# =========================
elif page == "🖼️ EDA 图表":
    st.title("🖼️ EDA 图表展示")

    eda_tabs = st.tabs(["综合分析", "词云图", "v3/v4 图表"])

    with eda_tabs[0]:
        st.subheader("v2 高级 EDA 图表")
        c1, c2 = st.columns(2)
        with c1:
            show_image(CHARTS_DIR / "01_restaurant_rating_review_bubble.png", "餐厅评分与热度分布")
            show_image(CHARTS_DIR / "03_cuisine_sentiment_rank.png", "各菜系好评率排序")
        with c2:
            show_image(CHARTS_DIR / "02_cuisine_performance_heatmap.png", "不同菜系综合表现对比")
            show_image(CHARTS_DIR / "04_price_rating_quadrant.png", "价格—评分性价比分层")

        show_image(CHARTS_DIR / "05_review_time_trend.png", "评论数量与平均评分时间趋势")

    with eda_tabs[1]:
        st.subheader("正负面评论词云")
        c1, c2 = st.columns(2)
        with c1:
            show_image(CHARTS_DIR / "wordcloud_positive.png", "正面评论词云")
        with c2:
            show_image(CHARTS_DIR / "wordcloud_negative.png", "负面评论词云")

        show_image(CHARTS_DIR / "wordcloud_positive_negative_compare.png", "正负面词云对比")

    with eda_tabs[2]:
        st.subheader("机器学习与深度学习相关图表")
        c1, c2 = st.columns(2)
        with c1:
            show_image(CHARTS_DIR / "v3_best_model_confusion_matrix.png", "v3 最优模型混淆矩阵")
            show_image(CHARTS_DIR / "v3_roc_curves.png", "v3 ROC 曲线")
        with c2:
            show_image(CHARTS_DIR / "v3_model_f1_comparison.png", "v3 模型 F1 对比")
            show_image(CHARTS_DIR / "ml_vs_dl.png", "ML vs DL 对比")


# =========================
# ML 情感预测
# =========================
elif page == "🤖 ML 情感预测":
    st.title("🤖 机器学习情感预测")

    st.info("该页面会尝试加载 v3 保存的 TF-IDF 向量器和 best_model.pkl。若环境版本不一致，可只展示模型结果。")

    results_path = MODELS_DIR / "model_results.csv"
    results_df = load_csv(str(results_path))

    if not results_df.empty:
        st.subheader("v3 模型结果")
        st.dataframe(results_df, use_container_width=True)
    else:
        st.warning("未找到 outputs/models/model_results.csv。")

    st.divider()

    tfidf, ml_model, msg = load_ml_assets()

    if msg:
        st.warning(msg)
        st.caption("展示型 v6 不强制要求实时预测，能展示结果表和图表即可。")
    else:
        st.success("ML 模型和 TF-IDF 向量器加载成功。")

        user_text = st.text_area(
            "输入一条餐厅评论：",
            value="这家店味道很好，分量很足，价格也实惠，下次还会再来。",
            height=120
        )

        if st.button("预测情感", type="primary"):
            text_for_model = simple_tokenize(user_text)
            X_vec = tfidf.transform([text_for_model])
            pred = ml_model.predict(X_vec)[0]
            label = sentiment_label(pred)

            if "正面" in label:
                st.success(f"预测结果：{label}")
            elif "负面" in label:
                st.error(f"预测结果：{label}")
            else:
                st.info(f"预测结果：{label}")

            with st.expander("查看模型输入文本"):
                st.write(text_for_model)

    st.subheader("模型评估图")
    c1, c2 = st.columns(2)
    with c1:
        show_image(CHARTS_DIR / "v3_best_model_confusion_matrix.png", "混淆矩阵")
    with c2:
        show_image(CHARTS_DIR / "v3_model_f1_comparison.png", "F1 对比")


# =========================
# DL 分析
# =========================
elif page == "🧠 DL 深度学习分析":
    st.title("🧠 深度学习分析")

    st.markdown(
        """
        本页面展示 v4 的 TextCNN 与 Bi-LSTM+Attention 训练结果。
        TextCNN 主要捕捉局部 n-gram 特征，Bi-LSTM+Attention 主要捕捉序列上下文并提供注意力可解释性。
        """
    )

    dl_results_path = MODELS_DIR / "ml_vs_dl_results.csv"
    dl_results = load_csv(str(dl_results_path))

    if not dl_results.empty:
        st.subheader("ML vs DL 结果表")
        st.dataframe(dl_results, use_container_width=True)
    else:
        st.warning("未找到 outputs/models/ml_vs_dl_results.csv。")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        show_image(CHARTS_DIR / "textcnn_training.png", "TextCNN 训练曲线")
    with c2:
        show_image(CHARTS_DIR / "bilstm_training.png", "Bi-LSTM 训练曲线")

    st.subheader("注意力可视化")
    show_image(CHARTS_DIR / "attention_visualization.png", "Bi-LSTM Attention 权重可视化")

    st.subheader("ML vs DL 对比图")
    show_image(CHARTS_DIR / "ml_vs_dl.png", "ML vs DL 情感分类性能对比")


# =========================
# RAG 结果
# =========================
elif page == "💬 RAG 问答结果":
    st.title("💬 RAG 问答结果展示")

    st.info("展示型 v6 不实时调用大模型，只展示 v5 已保存的无 RAG / RAG 对比结果和证据链。")

    no_rag_vs_rag = read_json(RAG_DIR / "no_rag_vs_rag.json")
    rag_experiments = read_json(RAG_DIR / "rag_experiments.json")

    tab1, tab2 = st.tabs(["无 RAG vs RAG", "参数实验"])

    with tab1:
        st.subheader("无 RAG 与 RAG 对比")

        if no_rag_vs_rag is None:
            st.warning("未找到 outputs/rag_results/no_rag_vs_rag.json。")
        else:
            st.markdown("**用户问题：**")
            st.write(no_rag_vs_rag.get("query", ""))

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("### 无 RAG 回答")
                st.write(no_rag_vs_rag.get("no_rag_answer", ""))

            with c2:
                st.markdown("### RAG 回答")
                st.write(no_rag_vs_rag.get("rag_answer", ""))

            st.markdown("### RAG 证据链")
            evidence = no_rag_vs_rag.get("evidence", [])
            if evidence:
                for e in evidence:
                    rank = e.get("rank", "")
                    title = f"证据 {rank}｜{e.get('restaurant_name', '')}｜{e.get('dish_name', '')}"
                    with st.expander(title):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("评分", e.get("rating", "-"))
                        c2.metric("价格", e.get("price", "-"))
                        c3.metric("相似度", f"{e.get('score', 0):.4f}" if "score" in e else "-")
                        st.write(e.get("review_text", ""))
            else:
                st.warning("JSON 中没有 evidence 字段或证据为空。")

    with tab2:
        st.subheader("top_k / temperature 参数实验")

        if rag_experiments is None:
            st.warning("未找到 outputs/rag_results/rag_experiments.json。")
        else:
            if isinstance(rag_experiments, list) and len(rag_experiments) > 0:
                labels = [
                    f"实验 {i+1}｜top_k={x.get('top_k', '-')}, temperature={x.get('temperature', '-')}"
                    for i, x in enumerate(rag_experiments)
                ]

                selected = st.selectbox("选择实验结果：", labels)
                idx = labels.index(selected)
                item = rag_experiments[idx]

                st.markdown("**问题：**")
                st.write(item.get("query", ""))

                st.markdown("**回答：**")
                st.write(item.get("answer", ""))

                st.markdown("**证据链：**")
                for e in item.get("evidence", []):
                    with st.expander(f"证据 {e.get('rank', '')}｜{e.get('restaurant_name', '')}｜{e.get('dish_name', '')}"):
                        st.write(f"评分：{e.get('rating', '-')}")
                        st.write(f"价格：{e.get('price', '-')}")
                        st.write(e.get("review_text", ""))
            else:
                st.warning("rag_experiments.json 不是有效列表或内容为空。")


# =========================
# 融合对比
# =========================
elif page == "📈 融合对比":
    st.title("📈 统计方法、机器学习、深度学习与 RAG 融合对比")

    st.markdown(
        """
        本页面用于展示不同阶段方法的作用差异：

        - **EDA**：回答“数据整体有什么规律”
        - **ML**：回答“给定评论能否自动分类”
        - **DL**：回答“深度模型是否带来更强表达能力”
        - **RAG**：回答“能否基于真实评论证据进行问答”
        """
    )

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("v3 机器学习结果")
        model_results = load_csv(str(MODELS_DIR / "model_results.csv"))
        if not model_results.empty:
            st.dataframe(model_results, use_container_width=True)
        else:
            st.warning("缺少 model_results.csv。")

    with c2:
        st.subheader("v4 ML vs DL 结果")
        ml_dl_results = load_csv(str(MODELS_DIR / "ml_vs_dl_results.csv"))
        if not ml_dl_results.empty:
            st.dataframe(ml_dl_results, use_container_width=True)
        else:
            st.warning("缺少 ml_vs_dl_results.csv。")

    st.subheader("对比图")
    c1, c2 = st.columns(2)
    with c1:
        show_image(CHARTS_DIR / "v3_roc_curves.png", "v3 ROC 曲线")
    with c2:
        show_image(CHARTS_DIR / "ml_vs_dl.png", "ML vs DL 对比")

    st.subheader("结论")
    st.markdown(
        """
        在当前模拟数据中，传统机器学习模型已经取得很高的分类效果，说明文本中的情感关键词与标签高度一致。
        深度学习模型完成了序列建模和注意力可视化，但在模板化数据上优势不明显。
        RAG 模块的核心价值不在分类准确率，而在于让大模型回答能够引用真实评论证据，从而降低幻觉风险。
        """
    )


# =========================
# 产物检查
# =========================
elif page == "📁 产物检查":
    st.title("📁 项目产物检查")

    required_files = {
        "v1 数据": [
            DATA_DIR / "processed_reviews.csv",
            DATA_DIR / "stopwords.txt",
            DATA_DIR / "userdict_restaurant.txt",
        ],
        "v2 图表": [
            CHARTS_DIR / "01_restaurant_rating_review_bubble.png",
            CHARTS_DIR / "02_cuisine_performance_heatmap.png",
            CHARTS_DIR / "03_cuisine_sentiment_rank.png",
            CHARTS_DIR / "04_price_rating_quadrant.png",
            CHARTS_DIR / "05_review_time_trend.png",
            CHARTS_DIR / "wordcloud_positive.png",
            CHARTS_DIR / "wordcloud_negative.png",
            CHARTS_DIR / "wordcloud_positive_negative_compare.png",
        ],
        "v3 模型": [
            MODELS_DIR / "tfidf_vectorizer.pkl",
            MODELS_DIR / "best_model.pkl",
            MODELS_DIR / "model_results.csv",
            CHARTS_DIR / "v3_best_model_confusion_matrix.png",
            CHARTS_DIR / "v3_model_f1_comparison.png",
            CHARTS_DIR / "v3_roc_curves.png",
        ],
        "v4 深度学习": [
            MODELS_DIR / "textcnn.pth",
            MODELS_DIR / "bilstm.pth",
            MODELS_DIR / "ml_vs_dl_results.csv",
            CHARTS_DIR / "textcnn_training.png",
            CHARTS_DIR / "bilstm_training.png",
            CHARTS_DIR / "attention_visualization.png",
            CHARTS_DIR / "ml_vs_dl.png",
        ],
        "v5 RAG": [
            RAG_DIR / "no_rag_vs_rag.json",
            RAG_DIR / "rag_experiments.json",
        ],
    }

    for group, files in required_files.items():
        st.subheader(group)
        rows = []
        for p in files:
            rows.append({
                "文件": str(p.relative_to(BASE_DIR)),
                "状态": "存在" if p.exists() else "缺失",
                "大小KB": round(p.stat().st_size / 1024, 2) if p.exists() else None
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.subheader("outputs/charts 下全部图片")
    if CHARTS_DIR.exists():
        chart_files = sorted([p.name for p in CHARTS_DIR.glob("*") if p.is_file()])
        st.write(chart_files)
    else:
        st.warning("outputs/charts 不存在。")
