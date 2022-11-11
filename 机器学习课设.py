# coding=gb2312
import pandas as pd
import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import Dataset
from sklearn.model_selection import train_test_split
#  1. 数据预处理
# 读取数据
df = pd.read_csv('train_dataset2.csv')
user_ids = df["user_id"].unique().tolist()
# 从新编码user 和 book，类似标签编码的过程
# 此步骤主要为减少id的编码空间
user2user_encoded = {x: i for i, x in enumerate(user_ids)}  # 编码键为x，值为i
userencoded2user = {i: x for i, x in enumerate(user_ids)}  # 编码键为i，值为x

book_ids = df["item_id"].unique().tolist()
book2book_encoded = {x: i for i, x in enumerate(book_ids)} # 编码键为x，值为i
book_encoded2book = {i: x for i, x in enumerate(book_ids)} # 编码键为i，值为x
# 编码映射
df["user"] = df["user_id"].map(user2user_encoded)  # 返回新dataframe
df["book"] = df["item_id"].map(book2book_encoded)  # 返回新dataframe
# 编码映射
df["user"] = df["user_id"].map(user2user_encoded)  # 返回新dataframe
df["book"] = df["item_id"].map(book2book_encoded)  # 返回新dataframe

num_users = len(user2user_encoded)  # 取用户总数
num_books = len(book_encoded2book)  # 取书籍总数

user_book_dict = df.iloc[:].groupby(['user'])['book'].apply(list)  # 取按用户和其评价过的书组成字典
# 随机挑选数据集作为负样本，负样本只需要对没有评价的书进行随机采样
neg_df = []  # 定义负样本
book_set = set(list(book_encoded2book.keys()))  # 取所有书的集合：将列表转化为集合,便于运算
for user_idx in user_book_dict.index:
    book_idx = book_set - set(list(user_book_dict.loc[user_idx]))  # 得到每个用户没评价过的书
    book_idx = list(book_idx)  # 转回list，list类型可以取随机数
    neg_book_idx = np.random.choice(book_idx, 100)  # 从用户没评价过的书中随机挑选100个样本
    for x in neg_book_idx:
        neg_df.append([user_idx, x])  # 将用户和样本组成列表

# 负样本的标签
neg_df = pd.DataFrame(neg_df, columns=['user', 'book'])   # 加上列名信息
neg_df['label'] = 0  # 加上标签
# 正样本的标签
df['label'] = 1
# 正负样本合并为数据集
train_df = pd.concat([df[['user', 'book', 'label']],
                      neg_df[['user', 'book', 'label']]], axis=0)  # 把数据集合并

train_df = train_df.sample(frac=1)  # 随机抽样


#  2. 准备训练和验证数据
# 自定义数据集
class SelfDefinedDataset(Dataset):
    def __init__(self, data_x, data_y, mode='train'):
        super(SelfDefinedDataset, self).__init__()
        self.data_x = data_x
        self.data_y = data_y
        self.mode = mode

        # 根据返回mode值返回样本数据或标签
    def __getitem__(self, idx):
        if self.mode == 'predict':
            return self.data_x[idx]
        else:
            return self.data_x[idx], self.data_y[idx]

    def __len__(self):
        return len(self.data_x)


# x_train：训练集数据 x_val：验证集数据 y_train：训练集标签 y_val：验证集标签
x_train, x_val, y_train, y_val = train_test_split(train_df[['user', 'book']].values,
train_df['label'].values.astype(np.float32).reshape(-1, 1))  # 随机将样本集合划分为训练集和验证集

traindataset = SelfDefinedDataset(x_train, y_train)  # 初始化训练集
train_loader = paddle.io.DataLoader(traindataset, batch_size=1280*4, shuffle=True)  # 定义并初始化训练集数据读取器

val_dataset = SelfDefinedDataset(x_val, y_val)  # 初始化验证集
val_loader = paddle.io.DataLoader(val_dataset, batch_size=1280*4, shuffle=True)  # 定义并初始化验证集数据读取器
# # 测试读取数据集
# for data, label in traindataset:
#     print(data.shape, label.shape)
#     print(data, label)
#     break

# # 测试dataloder读取
# for batch_id, data in enumerate(train_loader):
#     x_data = data[0]
#     y_data = data[1]
#
#     print(x_data.shape)
#     print(y_data.shape)
#     break

#  3.搭建模型
EMBEDDING_SIZE = 32
# 将用户和书嵌入到32维向量中。该模型计算用户和电影嵌入之间的匹配分数，并添加每部电影和每个用户的偏差。比赛分数通过 sigmoid 缩放到间隔[0, 1]。
class RecommenderNet(nn.Layer):
    def __init__(self, num_users, num_books, embedding_size):
        super(RecommenderNet, self).__init__()
        self.num_users = num_users
        self.num_books = num_books
        self.embedding_size = embedding_size
        # 创建一个用户属性对象
        weight_attr_user = paddle.ParamAttr(
            regularizer=paddle.regularizer.L2Decay(1e-6),  # 设置正则化规则
            initializer=nn.initializer.KaimingNormal()     # 设置初始化方式
        )
        self.user_embedding = nn.Embedding(
            num_users,
            embedding_size,
            weight_attr=weight_attr_user
        )
        self.user_bias = nn.Embedding(num_users, 1)
        weight_attr_book = paddle.ParamAttr(
            regularizer=paddle.regularizer.L2Decay(1e-6),
            initializer=nn.initializer.KaimingNormal()
        )
        self.book_embedding = nn.Embedding(
            num_books,
            embedding_size,
            weight_attr=weight_attr_book
        )
        self.book_bias = nn.Embedding(num_books, 1)

    def forward(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        book_vector = self.book_embedding(inputs[:, 1])
        book_bias = self.book_bias(inputs[:, 1])
        dot_user_book = paddle.dot(user_vector, book_vector)
        x = dot_user_book + user_bias + book_bias
        x = nn.functional.sigmoid(x)
        return x


#  4.模型训练和预测
# 模型的训练和验证
model = RecommenderNet(num_users, num_books, EMBEDDING_SIZE)
model = paddle.Model(model)  # 封装模型

# 定义模型损失函数、优化器和评价指标
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.001)  # 学习率默认0.001
loss = nn.BCELoss()
metric = paddle.metric.Precision()
# 设置日志保存路径
log_dir = './log'
callback = paddle.callbacks.VisualDL(log_dir=log_dir)
model.prepare(optimizer, loss, metric)  # 使用 Model.prepare 配置训练准备参数
model.fit(train_loader, val_loader, epochs=5,save_dir='./cc', verbose=1, callbacks=callback)
# 准备提交数据的格式
test_df = []
with open('submission.csv', 'w') as up:
    up.write('user_id,item_id\n')

# 模型预测
book_set = set(list(book_encoded2book.keys()))
for idx in range(int(len(user_book_dict) / 1000) + 1):
    # 对于所有的用户，需要预测他们未评价过但可能会喜欢的一本书
    test_user_idx = []
    test_book_idx = []
    for user_idx in user_book_dict.index[idx * 1000:(idx + 1) * 1000]:
        book_idx = book_set - set(list(user_book_dict.loc[user_idx]))
        book_idx = list(book_idx)
        test_user_idx += [user_idx] * len(book_idx)
        test_book_idx += book_idx

    # 从剩余书中筛选出标签为正的样本
    test_data = np.array([test_user_idx, test_book_idx]).T
    test_dataset = SelfDefinedDataset(test_data, data_y=None, mode='predict')
    test_loader = paddle.io.DataLoader(test_dataset, batch_size=1280, shuffle=False)

    test_predict = model.predict(test_loader, batch_size=1024)
    test_predict = np.concatenate(test_predict[0], 0)

    test_data = pd.DataFrame(test_data, columns=['user', 'book'])
    test_data['label'] = test_predict
    for gp in test_data.groupby(['user']):
        with open('submission.csv', 'a') as up:
            u = gp[0]
            b = gp[1]['book'].iloc[gp[1]['label'].argmax()]
            up.write(f'{userencoded2user[u]}, {book_encoded2book[b]}\n')
    del test_data, test_dataset, test_loader
print("运行结束")
