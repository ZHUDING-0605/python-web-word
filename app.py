import os
import random
import time
import streamlit as st
import requests
from bs4 import BeautifulSoup
import jieba
from collections import Counter
import pandas as pd
import re
from pyecharts import options as opts
from pyecharts.charts import WordCloud, Bar, Pie, Line, Scatter, HeatMap, Radar, Boxplot
import streamlit.components.v1 as components
from wordcloud import WordCloud as wc
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# 输入文章URL
st.title("文章词频分析")
url = st.text_input("请输入文章URL", "")


def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # 去除所有标点符号
    text = re.sub(r'\s+', '', text)  # 去除所有空格
    return text


if url:
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all('body')
        text = " ".join([para.get_text() for para in paragraphs])

        st.balloons()  # 气球等待特效
        with st.spinner('Wait for it...'):
            time.sleep(0.5)  # 等待时间

        clean_text_content = clean_text(text)  # 获取纯文本文章

        st.write("抓取到的文章内容：")
        st.text_area("文章内容", clean_text_content, height=300)  # 显示文章

        words = jieba.cut(clean_text_content)  # 使用 jieba.cut 对文本进行分词
        word_count = Counter(words)  # 使用 Counter 统计每个词的出现次数

        # 设置滑块，滑动值为最低词频词，默认为1
        min_freq = st.sidebar.slider("设置最低频词频", min_value=1, max_value=150, value=1)
        filtered_word_count = {word: count for word, count in word_count.items() if count >= min_freq}  # 筛选大于最小值的词频
        # 转换DataFrame类型，设置列名，按频率列降序排序
        word_freq_df = pd.DataFrame(filtered_word_count.items(), columns=["词语", "频率"]).sort_values(by="频率",
                                                                                                       ascending=False)
        st.write("词频排名前20的词汇：")
        st.dataframe(word_freq_df.head(20))

        # 词云绘制图片
        st.subheader("词云图")
        heart_mask_path = 'love.png'
        heart_mask = np.array(Image.open(heart_mask_path))

        # 生成词云图
        wordcloud = wc(
            font_path="msyh.ttc",  # 设置字体路径
            mask= heart_mask
            background_color="white",
            random_state=3,  # 设置有多少种随机生成状态，即有多少种配色方案
        ).generate_from_frequencies(filtered_word_count)

        # 显式创建一个fig, ax对象
        fig, ax = plt.subplots()  # 创建一个图形和坐标轴对象
        ax.imshow(wordcloud, interpolation="bilinear")  # 绘制词云图
        ax.axis("off")  # 关闭坐标轴
        # 使用st.pyplot显示图形
        st.pyplot(fig)

        # 创建图形选择框
        chart_type = st.sidebar.selectbox(
            "选择图形类型",
            ["柱状图", "饼图", "折线图", "散点图", "热力图", "面积图", "雷达图", "直方图"]
        )
        # 在创建前N个词频时，首先进行筛选
        top_n = st.sidebar.slider("显示前N个高频词", 1, 20, 20)  # 可以选择显示多少个词
        # 排序并过滤掉频率过低的词汇，按第二个值进行排序，降序排序
        top_n_filtered = {word: count for word, count in
                          sorted(filtered_word_count.items(), key=lambda item: item[1], reverse=True)}

        # 展示前top_n个词
        top_n_filtered = dict(list(top_n_filtered.items())[:top_n])

        # 显示前N个词汇的柱状图
        if chart_type == "柱状图":
            x_data = list(top_n_filtered.keys())
            y_data = list(top_n_filtered.values())  # 得到字典的所有键值
            bar_chart = (
                Bar()  # 初始化一个柱状图对象
                .add_xaxis(x_data)  # x轴显示前N个词
                .add_yaxis("频率", y_data)  # y轴显示频率
                # 设置全局的图表选项
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="词频柱状图"),  # 图表的标题为“词频柱状图”
                    xaxis_opts=opts.AxisOpts(name="词语", axislabel_opts=opts.LabelOpts(rotate=45)),
                    # 旋转x轴标签45度，以便更好地显示长的词汇。
                    yaxis_opts=opts.AxisOpts(name="频率"),
                )
                .set_series_opts(
                    label_opts=opts.LabelOpts(is_show=True, position="top")  # 在柱状图顶部显示频率值
                )

            )
            # 生成图表的 HTML 代码
            bar_chart_html = bar_chart.render_embed()
            # 将生成的 HTML 代码嵌入到 Streamlit 页面中
            components.html(bar_chart_html, height=600, width=1000)

        elif chart_type == "饼图":
            pie_chart = (
                Pie()  # 创建一个饼图对象
                .add("", [(k, v) for k, v in top_n_filtered.items()])  # 将词语和对应的频率作为数据添加到饼图中
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="词频饼图"),
                    legend_opts=opts.LegendOpts(is_show=True,
                                                orient='horizontal',  # 图例排列方式为横向
                                                pos_bottom='0%',  # 设置图例位置在图形下方
                                                pos_left='center'  # 将图例居中
                                                )  # 隐藏饼图的标识符
                )
                .set_series_opts(
                    label_opts=opts.LabelOpts(is_show=True, formatter="{b}: {c} ({d}%)")  # 显示百分比
                )
            )
            pie_chart_html = pie_chart.render_embed()
            components.html(pie_chart_html, height=600, width=1000)

        elif chart_type == "折线图":
            line_chart = (
                Line()
                .add_xaxis(list(top_n_filtered.keys()))  # 显示前N个词
                .add_yaxis("频率", list(top_n_filtered.values()))
                .set_global_opts(title_opts=opts.TitleOpts(title="词频折线图"))
            )
            line_chart_html = line_chart.render_embed()
            components.html(line_chart_html, height=600, width=1000)

        elif chart_type == "散点图":
            scatter_chart = (
                Scatter()
                .add_xaxis(list(top_n_filtered.keys()))
                .add_yaxis("频率", list(top_n_filtered.values()))
                .set_global_opts(title_opts=opts.TitleOpts(title="词频散点图"))
            )
            scatter_chart_html = scatter_chart.render_embed()
            components.html(scatter_chart_html, height=600, width=1000)


        elif chart_type == "热力图":
            heatmap_data = [[i, 0, freq] for i, freq in enumerate(top_n_filtered.values())]
            heatmap_chart = (
                HeatMap()
                .add_xaxis(list(top_n_filtered.keys()))  # x轴为词汇
                .add_yaxis("频率", ["频率"], heatmap_data)  # y轴为分类，单分类"频率"
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="词频热力图"),
                    visualmap_opts=opts.VisualMapOpts(max_=max(top_n_filtered.values())),  # 设置颜色映射范围
                    # max_ 为top_n_filtered 中频率值的最大值
                    xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=45)),  # 旋转x轴标签
                )
            )
            heatmap_chart_html = heatmap_chart.render_embed()
            components.html(heatmap_chart_html, height=600, width=1000)

        elif chart_type == "面积图":
            area_chart = (
                Line()  # 折线图对象
                .add_xaxis(list(top_n_filtered.keys()))
                .add_yaxis("频率", list(top_n_filtered.values()),
                           areastyle_opts=opts.AreaStyleOpts(opacity=0.5))  # 面积图的填充,填充的透明度为 50%
                .set_global_opts(title_opts=opts.TitleOpts(title="词频面积图"))
            )
            area_chart_html = area_chart.render_embed()
            components.html(area_chart_html, height=600, width=1000)

        elif chart_type == "雷达图":
            schema = [{"name": word, "max": max(top_n_filtered.values())} for word in top_n_filtered.keys()]
            radar_chart = (
                Radar()  # 初始化一个雷达图对象
                .add_schema(schema)  # 设置雷达图的维度schema,包含每个词汇和最大频率的字典列表
                .add("频率", [list(top_n_filtered.values())])
                .set_global_opts(title_opts=opts.TitleOpts(title="词频雷达图"))
            )
            radar_chart_html = radar_chart.render_embed()
            components.html(radar_chart_html, height=600, width=1000)


        elif chart_type == "直方图":
             # 设置直方图的区间大小（可以根据需要调整）
            bins = [i for i in range(0, max(top_n_filtered.values()) + 10, 10)]  # 词频划分为若干个 10 为一组的区间
            # 将词频数据分到各个区间
            freq_data = list(top_n_filtered.values())
            hist_data = [0] * len(bins)  # 初始时，所有区间的词汇数量都设置为 0

            # 将每个词频放入对应的区间
            for freq in freq_data:
                for i in range(len(bins) - 1):
                    if bins[i] <= freq < bins[i + 1]:
                        hist_data[i] += 1
                        break

            histogram_chart = (
                Bar()
                .add_xaxis([f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)])  # 区间范围
                .add_yaxis("词汇数", hist_data)  # 各个区间的词汇数量
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="词频直方图"),
                    xaxis_opts=opts.AxisOpts(name="频率区间"),
                    yaxis_opts=opts.AxisOpts(name="词汇数量"),
                )
            )
            histogram_chart_html = histogram_chart.render_embed()
            components.html(histogram_chart_html, height=600, width=1000)


    except Exception as e:
        st.error(f"错误信息：{str(e)}")
