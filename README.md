# 校园周边餐厅智能评价分析助手

本项目是一个面向校园周边餐厅评论数据的智能分析系统，覆盖了从数据预处理、探索性数据分析、机器学习分类、深度学习建模、大模型 RAG 问答到 Streamlit Web 应用集成的完整流程。

项目基于 5 万条校园周边餐厅评论数据，围绕“餐厅口碑分析、菜系表现比较、评论情感识别、用户关注点挖掘、基于评论证据的智能问答”等任务展开，最终形成一个可视化、可交互的餐厅智能评价分析助手。

---

## 一、项目功能概览

本项目按照 v1 到 v6 的顺序完成，每个版本对应一个技术阶段。

| 阶段 | 模块 | 主要内容 | 产出 |
|---|---|---|---|
| v1 | 数据预处理 | 数据清洗、去重、缺失值处理、中文分词、停用词过滤、自定义词典、情感标签构造 | `processed_reviews.csv`、分词对比、菜系高频词 |
| v2 | EDA 可视化 | 评分热度分析、菜系综合表现、好评率排序、价格评分四象限、时间趋势、词云分析 | 多张 EDA 图表、EDA 分析报告 |
| v3 | 机器学习分类 | TF-IDF 向量化，训练朴素贝叶斯、逻辑回归、SVM、随机森林、SGD 等模型 | `best_model.pkl`、`tfidf_vectorizer.pkl`、模型评估图 |
| v4 | 深度学习分类 | PyTorch 搭建 TextCNN、Bi-LSTM + Attention 情感分类模型 | `.pth` 模型、训练曲线、注意力可视化 |
| v5 | 大模型 RAG | 基于 Qwen2.5-3B-Instruct 构建 RAG 问答，完成无 RAG 与 RAG 对比 | `no_rag_vs_rag.json`、`rag_experiments.json` |
| v6 | 系统集成 | 使用 Streamlit 集成前五阶段成果，构建展示型 Web 应用 | `v6_day6_streamlit_app.py` |

---

## 二、项目亮点

1. **完整的数据分析链路**  
   从原始评论数据开始，逐步完成数据清洗、分词、特征工程、EDA、建模和系统展示。

2. **多模型情感分类对比**  
   同时使用传统机器学习模型和深度学习模型进行评论情感分类，并比较不同模型的效果和训练成本。

3. **基于真实评论证据的 RAG 问答**  
   通过检索餐厅评论数据，为大模型提供真实证据，降低大模型直接回答时的幻觉风险。

4. **Streamlit 展示型系统集成**  
   将 EDA 图表、模型结果、深度学习分析和 RAG 问答结果统一集成到 Web 页面中，便于展示和答辩。

---

## 三、项目目录结构

```text
restaurant-review-ai-assistant/
├── data/
│   ├── processed_reviews.csv              # v1 预处理后的评论数据
│   ├── stopwords.txt                      # 停用词表
│   └── userdict_restaurant.txt            # 餐饮领域自定义词典
│
├── outputs/
│   ├── charts/                            # v2/v3/v4 生成的图表
│   │   ├── 01_restaurant_rating_review_bubble.png
│   │   ├── 02_cuisine_performance_heatmap.png
│   │   ├── 03_cuisine_sentiment_rank.png
│   │   ├── 04_price_rating_quadrant.png
│   │   ├── 05_review_time_trend.png
│   │   ├── wordcloud_positive.png
│   │   ├── wordcloud_negative.png
│   │   ├── wordcloud_positive_negative_compare.png
│   │   ├── v3_best_model_confusion_matrix.png
│   │   ├── v3_model_f1_comparison.png
│   │   ├── v3_roc_curves.png
│   │   ├── textcnn_training.png
│   │   ├── bilstm_training.png
│   │   ├── attention_visualization.png
│   │   └── ml_vs_dl.png
│   │
│   ├── models/                            # v3/v4 模型与结果文件
│   │   ├── best_model.pkl
│   │   ├── tfidf_vectorizer.pkl
│   │   ├── model_results.csv
│   │   ├── cross_validation_results.csv
│   │   ├── gridsearch_lr_results.csv
│   │   └── ml_vs_dl_results.csv
│   │
│   ├── rag_results/                       # v5 RAG 问答结果
│   │   ├── no_rag_vs_rag.json
│   │   └── rag_experiments.json
│   │
│   └── v1/
│       ├── jieba_dict_comparison.csv
│       └── cuisine_top10_words.csv
│
├── v6_day6_streamlit_app.py               # Streamlit 展示型 Web 应用
├── v1_v2_v3_analysis.ipynb                # 数据处理、EDA、机器学习 Notebook
├── v4_deep_learning.ipynb                 # 深度学习 Notebook
├── v5_qwen_rag.ipynb                      # 大模型 RAG Notebook
├── requirements.txt                       # 项目依赖
├── README.md
└── .gitignore
