# 对比文本分类不同训练方法对比
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, LSTM, Dropout
from transformers import BertTokenizer, TFBertForSequenceClassification

# 设置随机种子保证可复现性
np.random.seed(42)
tf.random.set_seed(42)

# ====================== 1. 数据加载与预处理 ======================
print("正在加载数据集...")
# 使用20个新闻组数据集（简化为二分类任务：科技 vs 娱乐）
categories = ['sci.space', 'rec.motorcycles']
newsgroups = fetch_20newsgroups(subset='all', categories=categories, 
                                remove=('headers', 'footers', 'quotes'),
                                random_state=42)

X = newsgroups.data
y = newsgroups.target
class_names = newsgroups.target_names

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"数据集大小: 训练集 {len(X_train)}, 测试集 {len(X_test)}")
print(f"类别分布: {class_names[0]}: {np.sum(y==0)}, {class_names[1]}: {np.sum(y==1)}")

# ====================== 2. 定义评估函数 ======================
def evaluate_model(y_true, y_pred, model_name, train_time):
    """评估模型性能并返回结果字典"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\n{'='*50}")
    print(f"{model_name} 评估结果")
    print(f"{'='*50}")
    print(f"训练时间: {train_time:.2f} 秒")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return {
        '模型名称': model_name,
        '训练时间(秒)': round(train_time, 2),
        '准确率': round(accuracy, 4),
        '精确率': round(precision, 4),
        '召回率': round(recall, 4),
        'F1分数': round(f1, 4)
    }

# ====================== 3. 方法1: TF-IDF + 逻辑回归 ======================
print("\n\n正在训练 TF-IDF + 逻辑回归 模型...")
start_time = time.time()

# 文本向量化
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 训练模型
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)

# 预测
lr_pred = lr_model.predict(X_test_tfidf)
train_time = time.time() - start_time

# 评估
lr_results = evaluate_model(y_test, lr_pred, "TF-IDF + 逻辑回归", train_time)

# ====================== 4. 方法2: TF-IDF + 朴素贝叶斯 ======================
print("\n\n正在训练 TF-IDF + 朴素贝叶斯 模型...")
start_time = time.time()

# 训练模型
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

# 预测
nb_pred = nb_model.predict(X_test_tfidf)
train_time = time.time() - start_time

# 评估
nb_results = evaluate_model(y_test, nb_pred, "TF-IDF + 朴素贝叶斯", train_time)

# ====================== 5. 方法3: CNN 文本分类 ======================
print("\n\n正在训练 CNN 文本分类模型...")
start_time = time.time()

# 文本预处理
max_words = 10000
max_len = 200

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# 构建CNN模型
cnn_model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# 训练模型
history = cnn_model.fit(X_train_pad, y_train,
                        epochs=5,
                        batch_size=32,
                        validation_split=0.1,
                        verbose=1)

# 预测
cnn_pred_proba = cnn_model.predict(X_test_pad, verbose=0)
cnn_pred = (cnn_pred_proba > 0.5).astype(int).flatten()
train_time = time.time() - start_time

# 评估
cnn_results = evaluate_model(y_test, cnn_pred, "CNN 文本分类", train_time)

# ====================== 6. 方法4: LSTM 文本分类 ======================
print("\n\n正在训练 LSTM 文本分类模型...")
start_time = time.time()

# 构建LSTM模型
lstm_model = Sequential([
    Embedding(max_words, 128, input_length=max_len),
    LSTM(64, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

# 训练模型
history = lstm_model.fit(X_train_pad, y_train,
                         epochs=5,
                         batch_size=32,
                         validation_split=0.1,
                         verbose=1)

# 预测
lstm_pred_proba = lstm_model.predict(X_test_pad, verbose=0)
lstm_pred = (lstm_pred_proba > 0.5).astype(int).flatten()
train_time = time.time() - start_time

# 评估
lstm_results = evaluate_model(y_test, lstm_pred, "LSTM 文本分类", train_time)

# ====================== 7. 方法5: BERT 预训练模型 ======================
print("\n\n正在训练 BERT 预训练模型...")
start_time = time.time()

# 加载BERT分词器和模型
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 文本编码
def bert_encode(texts, tokenizer, max_len=128):
    encodings = tokenizer(texts,
                          truncation=True,
                          padding='max_length',
                          max_length=max_len,
                          return_tensors='tf')
    return encodings

train_encodings = bert_encode(X_train, bert_tokenizer)
test_encodings = bert_encode(X_test, bert_tokenizer)

# 转换为TensorFlow数据集
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    y_train
)).shuffle(1000).batch(16)

test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
    y_test
)).batch(16)

# 编译模型
bert_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])

# 训练模型
history = bert_model.fit(train_dataset,
                         epochs=3,
                         validation_data=test_dataset,
                         verbose=1)

# 预测
bert_pred_logits = bert_model.predict(test_dataset, verbose=0).logits
bert_pred = tf.argmax(bert_pred_logits, axis=1).numpy()
train_time = time.time() - start_time

# 评估
bert_results = evaluate_model(y_test, bert_pred, "BERT 预训练模型", train_time)

# ====================== 8. 结果汇总与可视化 ======================
print("\n\n" + "="*70)
print("所有模型效果对比汇总")
print("="*70)

# 汇总所有结果
results_df = pd.DataFrame([lr_results, nb_results, cnn_results, lstm_results, bert_results])
print(results_df.to_string(index=False))

# 可视化对比
plt.figure(figsize=(15, 10))

# 准确率对比
plt.subplot(2, 2, 1)
sns.barplot(x='模型名称', y='准确率', data=results_df, palette='viridis')
plt.title('各模型准确率对比')
plt.xticks(rotation=45, ha='right')
plt.ylim(0.8, 1.0)

# F1分数对比
plt.subplot(2, 2, 2)
sns.barplot(x='模型名称', y='F1分数', data=results_df, palette='viridis')
plt.title('各模型F1分数对比')
plt.xticks(rotation=45, ha='right')
plt.ylim(0.8, 1.0)

# 训练时间对比
plt.subplot(2, 2, 3)
sns.barplot(x='模型名称', y='训练时间(秒)', data=results_df, palette='rocket')
plt.title('各模型训练时间对比')
plt.xticks(rotation=45, ha='right')

# 综合性能散点图（准确率 vs 训练时间）
plt.subplot(2, 2, 4)
sns.scatterplot(x='训练时间(秒)', y='准确率', hue='模型名称', 
                data=results_df, s=200, palette='deep')
plt.title('准确率 vs 训练时间')
for i, row in results_df.iterrows():
    plt.text(row['训练时间(秒)']+2, row['准确率'], row['模型名称'], fontsize=9)

plt.tight_layout()
plt.show()

# 保存结果到CSV
results_df.to_csv('text_classification_comparison.csv', index=False)
print("\n结果已保存到 text_classification_comparison.csv")

