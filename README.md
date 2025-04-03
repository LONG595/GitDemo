# 基于多项式朴素贝叶斯的中文邮件分类系统

## 系统概述

本系统旨在高效地对中文邮件进行自动分类，能够准确区分垃圾邮件和普通邮件。系统采用多项式朴素贝叶斯算法作为核心分类技术，通过分析邮件文本内容中的词汇特征，实现智能判断。该系统具有高效性、准确性和灵活性，适用于大规模邮件分类任务。

## 核心功能

1. **邮件分类**：自动将邮件分为垃圾邮件和普通邮件。
2. **特征提取**：支持高频词特征和TF-IDF特征两种模式。
3. **样本平衡**：集成SMOTE算法，处理数据不平衡问题。
4. **模型评估**：提供详细的分类报告，包括精确率、召回率和F1分数。

## 算法说明

本系统采用多项式朴素贝叶斯分类器，具有以下特点：

1. **特征独立性假设**：假设每个特征（单词）在给定类别条件下相互独立，这大大简化了概率计算过程。
2. **概率计算机制**：通过计算单词在不同类别中出现的条件概率来进行分类预测。
3. **多项式分布**：专门处理离散型特征（如词频），适合文本分类场景。

## 数据处理流程

### 1. 文本清洗

系统首先对原始邮件文本进行深度清洗，去除标点、数字等干扰字符：
```python
import re
# 使用正则表达式去除标点、数字等干扰字符
line = re.sub(r'[^\w\s]', '', line)
```

### 2. 中文分词处理

采用jieba分词工具进行精准的中文分词：
```python
import jieba
# 执行分词并过滤无效词汇
words = jieba.cut(line)  # jieba分词
filtered_words = [word for word in words if len(word) > 1]  # 过滤单字词
```

### 3. 停用词过滤

去除常见停用词，以减少噪声：
```python
# 加载停用词列表
stop_words = set()
with open('stopwords.txt', 'r', encoding='utf-8') as f:
    for word in f:
        stop_words.add(word.strip())

# 过滤停用词
filtered_words = [word for word in filtered_words if word not in stop_words]
```

### 4. 样本平衡处理

针对数据不平衡问题，系统集成了SMOTE过采样技术：
```python
from imblearn.over_sampling import SMOTE
# 使用SMOTE算法平衡样本分布
smote = SMOTE(random_state=42)
vector_resampled, labels_resampled = smote.fit_resample(vector, labels)
```

## 特征工程

### 高频词特征模式（默认）

系统默认采用高频词特征提取方式：
```python
# 统计特征词出现频次构建特征向量
word_map = list(map(lambda word: words.count(word), top_words))
vector.append(word_map)
```

### TF-IDF特征模式（可选）

也可切换至TF-IDF加权特征模式：
```python
from sklearn.feature_extraction.text import TfidfVectorizer
# 使用TF-IDF加权构建特征向量
vectorizer = TfidfVectorizer(vocabulary=top_words)
vector = vectorizer.fit_transform([" ".join(words) for words in all_words])
```

## 模型训练与评估

### 模型训练

使用多项式朴素贝叶斯分类器进行训练：
```python
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train, y_train)
```

### 模型评估

系统建立了完善的评估机制：
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 数据划分保持原始分布
X_train, X_test, y_train, y_test = train_test_split(
    vector, labels, test_size=0.2, random_state=42, stratify=labels)

# 生成详细分类报告
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['普通邮件', '垃圾邮件']))
```

## 代码运行结果

### 默认分类模式，对应代码`classify.py`
<img src="C:\Users\21101\PycharmProjects\NLP\images\屏幕截图 2025-04-03 140215.png" width="800" alt="classify">

### 灵活切换方式
#### 局部切换 对应代码classify_local.py
<img src="C:\Users\21101\PycharmProjects\NLP\images\屏幕截图 2025-04-03 142201.png" width="800" alt="local">

#### 局部切换 对应代码classify_global.py
<img src="C:\Users\21101\PycharmProjects\NLP\images\屏幕截图 2025-04-03 142132.png" width="800" alt="global">

### 样本平衡处理
<img src="C:\Users\21101\PycharmProjects\NLP\images\屏幕截图 2025-04-03 142100.png" width="800" alt="sample_balancing">

### 最终版_添加全局方法选择/样本平衡处理/模型评估指标
<img src="C:\Users\21101\PycharmProjects\NLP\images\屏幕截图 2025-04-03 142241.png" width="800" alt="classify_all">
