# coding=gb2312
import pandas as pd
import numpy as np
import paddle
import paddle.nn as nn
from paddle.io import Dataset
from sklearn.model_selection import train_test_split
#  1. ����Ԥ����
# ��ȡ����
df = pd.read_csv('train_dataset2.csv')
user_ids = df["user_id"].unique().tolist()
# ���±���user �� book�����Ʊ�ǩ����Ĺ���
# �˲�����ҪΪ����id�ı���ռ�
user2user_encoded = {x: i for i, x in enumerate(user_ids)}  # �����Ϊx��ֵΪi
userencoded2user = {i: x for i, x in enumerate(user_ids)}  # �����Ϊi��ֵΪx

book_ids = df["item_id"].unique().tolist()
book2book_encoded = {x: i for i, x in enumerate(book_ids)} # �����Ϊx��ֵΪi
book_encoded2book = {i: x for i, x in enumerate(book_ids)} # �����Ϊi��ֵΪx
# ����ӳ��
df["user"] = df["user_id"].map(user2user_encoded)  # ������dataframe
df["book"] = df["item_id"].map(book2book_encoded)  # ������dataframe
# ����ӳ��
df["user"] = df["user_id"].map(user2user_encoded)  # ������dataframe
df["book"] = df["item_id"].map(book2book_encoded)  # ������dataframe

num_users = len(user2user_encoded)  # ȡ�û�����
num_books = len(book_encoded2book)  # ȡ�鼮����

user_book_dict = df.iloc[:].groupby(['user'])['book'].apply(list)  # ȡ���û��������۹���������ֵ�
# �����ѡ���ݼ���Ϊ��������������ֻ��Ҫ��û�����۵�������������
neg_df = []  # ���帺����
book_set = set(list(book_encoded2book.keys()))  # ȡ������ļ��ϣ����б�ת��Ϊ����,��������
for user_idx in user_book_dict.index:
    book_idx = book_set - set(list(user_book_dict.loc[user_idx]))  # �õ�ÿ���û�û���۹�����
    book_idx = list(book_idx)  # ת��list��list���Ϳ���ȡ�����
    neg_book_idx = np.random.choice(book_idx, 100)  # ���û�û���۹������������ѡ100������
    for x in neg_book_idx:
        neg_df.append([user_idx, x])  # ���û�����������б�

# �������ı�ǩ
neg_df = pd.DataFrame(neg_df, columns=['user', 'book'])   # ����������Ϣ
neg_df['label'] = 0  # ���ϱ�ǩ
# �������ı�ǩ
df['label'] = 1
# ���������ϲ�Ϊ���ݼ�
train_df = pd.concat([df[['user', 'book', 'label']],
                      neg_df[['user', 'book', 'label']]], axis=0)  # �����ݼ��ϲ�

train_df = train_df.sample(frac=1)  # �������


#  2. ׼��ѵ������֤����
# �Զ������ݼ�
class SelfDefinedDataset(Dataset):
    def __init__(self, data_x, data_y, mode='train'):
        super(SelfDefinedDataset, self).__init__()
        self.data_x = data_x
        self.data_y = data_y
        self.mode = mode

        # ���ݷ���modeֵ�����������ݻ��ǩ
    def __getitem__(self, idx):
        if self.mode == 'predict':
            return self.data_x[idx]
        else:
            return self.data_x[idx], self.data_y[idx]

    def __len__(self):
        return len(self.data_x)


# x_train��ѵ�������� x_val����֤������ y_train��ѵ������ǩ y_val����֤����ǩ
x_train, x_val, y_train, y_val = train_test_split(train_df[['user', 'book']].values,
train_df['label'].values.astype(np.float32).reshape(-1, 1))  # ������������ϻ���Ϊѵ��������֤��

traindataset = SelfDefinedDataset(x_train, y_train)  # ��ʼ��ѵ����
train_loader = paddle.io.DataLoader(traindataset, batch_size=1280*4, shuffle=True)  # ���岢��ʼ��ѵ�������ݶ�ȡ��

val_dataset = SelfDefinedDataset(x_val, y_val)  # ��ʼ����֤��
val_loader = paddle.io.DataLoader(val_dataset, batch_size=1280*4, shuffle=True)  # ���岢��ʼ����֤�����ݶ�ȡ��
# # ���Զ�ȡ���ݼ�
# for data, label in traindataset:
#     print(data.shape, label.shape)
#     print(data, label)
#     break

# # ����dataloder��ȡ
# for batch_id, data in enumerate(train_loader):
#     x_data = data[0]
#     y_data = data[1]
#
#     print(x_data.shape)
#     print(y_data.shape)
#     break

#  3.�ģ��
EMBEDDING_SIZE = 32
# ���û�����Ƕ�뵽32ά�����С���ģ�ͼ����û��͵�ӰǶ��֮���ƥ������������ÿ����Ӱ��ÿ���û���ƫ���������ͨ�� sigmoid ���ŵ����[0, 1]��
class RecommenderNet(nn.Layer):
    def __init__(self, num_users, num_books, embedding_size):
        super(RecommenderNet, self).__init__()
        self.num_users = num_users
        self.num_books = num_books
        self.embedding_size = embedding_size
        # ����һ���û����Զ���
        weight_attr_user = paddle.ParamAttr(
            regularizer=paddle.regularizer.L2Decay(1e-6),  # �������򻯹���
            initializer=nn.initializer.KaimingNormal()     # ���ó�ʼ����ʽ
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


#  4.ģ��ѵ����Ԥ��
# ģ�͵�ѵ������֤
model = RecommenderNet(num_users, num_books, EMBEDDING_SIZE)
model = paddle.Model(model)  # ��װģ��

# ����ģ����ʧ�������Ż���������ָ��
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.001)  # ѧϰ��Ĭ��0.001
loss = nn.BCELoss()
metric = paddle.metric.Precision()
# ������־����·��
log_dir = './log'
callback = paddle.callbacks.VisualDL(log_dir=log_dir)
model.prepare(optimizer, loss, metric)  # ʹ�� Model.prepare ����ѵ��׼������
model.fit(train_loader, val_loader, epochs=5,save_dir='./cc', verbose=1, callbacks=callback)
# ׼���ύ���ݵĸ�ʽ
test_df = []
with open('submission.csv', 'w') as up:
    up.write('user_id,item_id\n')

# ģ��Ԥ��
book_set = set(list(book_encoded2book.keys()))
for idx in range(int(len(user_book_dict) / 1000) + 1):
    # �������е��û�����ҪԤ������δ���۹������ܻ�ϲ����һ����
    test_user_idx = []
    test_book_idx = []
    for user_idx in user_book_dict.index[idx * 1000:(idx + 1) * 1000]:
        book_idx = book_set - set(list(user_book_dict.loc[user_idx]))
        book_idx = list(book_idx)
        test_user_idx += [user_idx] * len(book_idx)
        test_book_idx += book_idx

    # ��ʣ������ɸѡ����ǩΪ��������
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
print("���н���")
