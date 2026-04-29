import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader

# 超参数设置
# 随机种子，保证实验可复现
SEED = 42
# 输入句子固定长度：5个字
MAX_SENTENCE_LENGTH = 5
# 词嵌入向量维度
EMBEDDING_DIMENSION = 64
# RNN/LSTM隐藏层维度
HIDDEN_DIMENSION = 128
# 训练批次大小
BATCH_SIZE = 32
# 学习率
LEARNING_RATE = 0.001
# 训练轮数
TRAIN_EPOCHS = 10
# Dropout失活概率，防止过拟合
DROPOUT_RATE = 0.3
# 固定Python随机种子
random.seed(SEED)
# 固定PyTorch随机种子
torch.manual_seed(SEED)


# 数据生成函数
def generate_ni_position_sample():
    """
    生成一条5字样本，包含且仅包含一个"你"
    返回：句子字符串，标签（0-4对应位置1-5）
    """
    # 定义随机选用的汉字列表
    char_list = [ch for ch in "我他她它你们这那好坏美丑开心高低克一三五期间阿萨德"]
    # 随机生成"你"所在位置 0~4
    ni_position = random.randint(0, 4)
    # 随机生成5个汉字
    sentence = [random.choice(char_list) for _ in range(MAX_SENTENCE_LENGTH)]
    # 将指定位置替换为"你"
    sentence[ni_position] = "你"
    # 返回句子和标签
    return "".join(sentence), ni_position


# 生成10000条测试数据
total_sample_data = [generate_ni_position_sample() for _ in range(10000)]


# 构建字符词表
def build_character_vocab(sample_data):
    """
    根据所有文本构建字符级词表
    返回：字典{字符:编号}
    """
    # 初始化词表，0填充，1未知字符
    vocab_dict = {"<PAD>": 0, "<UNK>": 1}
    # 遍历每一条句子
    for sentence, _ in sample_data:
        # 遍历句子每个字
        for char in sentence:
            # 字不在词表中则添加
            if char not in vocab_dict:
                vocab_dict[char] = len(vocab_dict)
    return vocab_dict


# 构建完成的词表
character_vocab = build_character_vocab(total_sample_data)


# 超参数设置
def sentence_to_ids(sentence):
    """将句子文字转换为词表编号"""
    # 不存在的字用1（UNK）表示
    return [character_vocab.get(char, 1) for char in sentence]


# 自定义数据集类
class NiPositionDataset(Dataset):
    def __init__(self, dataset_data):
        """初始化：文本转编号，标签单独存储"""
        self.input_ids = [sentence_to_ids(sent) for sent, _ in dataset_data]
        self.labels = [label for _, label in dataset_data]

    def __len__(self):
        """返回数据总数"""
        return len(self.labels)

    def __getitem__(self, index):
        """根据索引获取一条数据（张量格式）"""
        return torch.LongTensor(self.input_ids[index]), torch.LongTensor([self.labels[index]]).squeeze()


# RNN模型
class NiPositionRNNClassifier(nn.Module):
    """RNN文本分类模型"""
    def __init__(self):
        # 调用父类初始化
        super().__init__()
        # 词嵌入层：编号 → 向量
        self.embedding_layer = nn.Embedding(len(character_vocab), EMBEDDING_DIMENSION, padding_idx=0)
        # RNN层
        self.rnn_layer = nn.RNN(input_size=EMBEDDING_DIMENSION, hidden_size=HIDDEN_DIMENSION, batch_first=True)
        # Dropout层
        self.dropout_layer = nn.Dropout(DROPOUT_RATE)
        # 全连接分类层，输出5分类
        self.classifier_layer = nn.Linear(HIDDEN_DIMENSION, 5)

    def forward(self, x, y=None):
        """
        前向传播
        x: 输入编号
        y: 标签（可选）
        如果传入y → 返回交叉熵损失
        不传入y → 返回softmax概率
        """
        # 词嵌入：[B,5] → [B,5,64]
        x = self.embedding_layer(x)
        # RNN计算
        out, _ = self.rnn_layer(x)
        # 取最后一步时间步的输出作为句子特征
        feat = out[:, -1, :]
        # Dropout正则化
        feat = self.dropout_layer(feat)
        # 全连接层得到原始输出分数
        out = self.classifier_layer(feat)

        # 判断是否传入标签
        if y is not None:
            # 传入标签：返回交叉熵损失
            return nn.functional.cross_entropy(out, y)
        else:
            # 无标签：返回softmax概率分布
            return torch.softmax(out, dim=1)


# LSTM 模型
class NiPositionLSTMClassifier(nn.Module):
    """LSTM文本分类模型"""
    def __init__(self):
        super().__init__()
        # 词嵌入层
        self.embedding_layer = nn.Embedding(len(character_vocab), EMBEDDING_DIMENSION, padding_idx=0)
        # LSTM层
        self.lstm_layer = nn.LSTM(input_size=EMBEDDING_DIMENSION, hidden_size=HIDDEN_DIMENSION,batch_first=True)
        # Dropout层
        self.dropout_layer = nn.Dropout(DROPOUT_RATE)
        # 分类层
        self.classifier_layer = nn.Linear(HIDDEN_DIMENSION, 5)

    def forward(self, x, y=None):
        """
        前向传播
        x: 输入
        y: 标签（可选）
        有y → 损失
        无y → 概率
        """
        # 词嵌入
        x = self.embedding_layer(x)
        # LSTM计算
        out, _ = self.lstm_layer(x)
        # 取最后一步特征
        feat = out[:, -1, :]
        # Dropout
        feat = self.dropout_layer(feat)
        # 原始输出
        out = self.classifier_layer(feat)

        # 判断是否计算损失
        if y is not None:
            # 传入标签，返回交叉熵损失
            return nn.functional.cross_entropy(out, y)
        else:
            # 不传入标签，返回softmax概率
            return torch.softmax(out, dim=1)


# 训练与评估函数
def train_evaluate_model(model, model_name):
    # 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 按8:2划分训练集、测试集
    split_index = int(0.8 * len(total_sample_data))
    train_dataset = total_sample_data[:split_index]
    test_dataset = total_sample_data[split_index:]

    # 构建数据加载器
    train_data_loader = DataLoader(NiPositionDataset(train_dataset), batch_size=BATCH_SIZE, shuffle=True)
    test_data_loader = DataLoader(NiPositionDataset(test_dataset), batch_size=BATCH_SIZE, shuffle=False)

    # 打印训练开始信息
    print(f"\n========== 开始训练 {model_name} ==========")
    # 开始逐轮训练
    for epoch in range(TRAIN_EPOCHS):
        # 开启训练模式
        model.train()
        # 记录本轮总损失
        total_epoch_loss = 0

        # 遍历训练批次
        for batch_input_ids, batch_labels in train_data_loader:
            # 梯度清零
            optimizer.zero_grad()
            # 传入输入+标签 → 模型直接返回损失
            loss = model(batch_input_ids, batch_labels)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 累计损失
            total_epoch_loss += loss.item()

        # 开启评估模式
        model.eval()
        # 正确预测数量
        correct_predict = 0
        # 总预测数量
        total_predict = 0

        # 评估时不计算梯度
        with torch.no_grad():
            for batch_input_ids, batch_labels in test_data_loader:
                # 只传输入 → 返回概率
                prob = model(batch_input_ids)
                # 取概率最大的类别
                predict_class = prob.argmax(dim=1)
                # 统计正确数
                correct_predict += (predict_class == batch_labels).sum().item()
                # 统计总数
                total_predict += batch_labels.size(0)

        # 计算测试集准确率
        test_accuracy = correct_predict / total_predict
        # 打印日志
        print(f"Epoch {epoch + 1:2d} | loss: {total_epoch_loss / len(train_data_loader):.4f} | test_acc: "
              f"{test_accuracy:.4f}")

    # ===================== 推理演示 =====================
    print(f"\n{model_name} 推理演示：")
    # 演示句子
    demo_sentences = ["你好世界啊", "我你今天很", "今天你开心", "我很你快乐", "我很开心你"]
    model.eval()
    # 逐条预测
    for sentence in demo_sentences:
        # 转换为模型输入张量
        input_tensor = torch.LongTensor([sentence_to_ids(sentence)])
        # 模型返回概率
        prob = model(input_tensor)
        # 获取预测位置
        predict_position = prob.argmax(1).item()
        # 打印结果
        print(f"输入语句:{sentence}, 预测'你'在第 {predict_position + 1} 位")


if __name__ == "__main__":
    # 训练RNN模型
    train_evaluate_model(NiPositionRNNClassifier(), "RNN模型")
    # 训练LSTM模型
    train_evaluate_model(NiPositionLSTMClassifier(), "LSTM模型")
