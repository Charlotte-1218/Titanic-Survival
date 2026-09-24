import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

SEED = 921
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

df = pd.read_csv('../train.csv')

# 统计缺失数量
print("——每列缺失值的个数——")
print(df.isnull().sum())

# 处理缺失值
df = df.drop(columns=['Cabin'])

train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=924)

age_median = df.loc[train_idx, 'Age'].median()
embarked_mode = df.loc[train_idx, 'Embarked'].mode()[0]
df.loc[train_idx, 'Age'] = df.loc[train_idx, 'Age'].fillna(age_median)
df.loc[test_idx, 'Age'] = df.loc[test_idx, 'Age'].fillna(age_median)
df.loc[train_idx, 'Embarked'] = df.loc[train_idx, 'Embarked'].fillna(embarked_mode)
df.loc[test_idx, 'Embarked'] = df.loc[test_idx, 'Embarked'].fillna(embarked_mode)

# 特征工程
df = df.drop(columns=['PassengerId', 'Name', 'Ticket'])# 无规律，删除
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})# 分清0和1！
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)# 独热编码

# 分离特征和结果
x = df.drop('Survived', axis=1).astype(float).values
y = df['Survived'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=924)

# 标准化
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std
np.savez('../preprocess_params.npz', mean=mean, std=std)

print("——数据预处理完毕，训练集大小：", x_train.shape, "测试集大小：", x_test.shape)

class TitanicDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32).view(-1,1)

    def __len__(self):
            return len(self.y)

    def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

train_loader = DataLoader(TitanicDataset(x_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TitanicDataset(x_test, y_test), batch_size=32, shuffle=False)

class TitanicModel(nn.Module):
    def __init__(self, input_dim):
        super(TitanicModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = TitanicModel(x_train.shape[1])
print("\n——模型结构已搭建——")
print(model)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100
train_losses = []
test_accuracies = []
train_accuracies = []

print("\n=== 开始训练 ===")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))

    model.eval()
    correct_train,total_train = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in train_loader:
            outputs = model(batch_x)
            predicted = (outputs > 0.5).float()
            total_train += batch_y.size(0)
            correct_train += (predicted == batch_y).sum().item()
    train_acc = correct_train / total_train
    train_accuracies.append(train_acc)

    correct, total = 0, 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            predicted = (outputs > 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    acc = correct / total
    test_accuracies.append(acc)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {train_losses[-1]:.4f}, Test Acc: {acc:.4f}")

torch.save(model.state_dict(), '../titanic_model.pth')
print("\n=== 模型已保存为 titanic_model.pth ===")

def predict_new_passenger(passenger_data):
    params = np.load('../preprocess_params.npz')
    mean, std = params['mean'], params['std']

    input_data = np.array([passenger_data])
    input_data = (input_data - mean) / std

    model.eval()
    with torch.no_grad():
        tensor_input = torch.tensor(input_data, dtype=torch.float32)
        prob = model(tensor_input).item()

        print(f"该乘客幸存概率: {prob:.4f}")
        return 1 if prob > 0.5 else 0

test_passenger = [3, 0, 22, 1, 0, 7.25, 0, 0]
result = predict_new_passenger(test_passenger)
print(f"最终预测结果: {result} (0=遇难, 1=幸存)")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses,label='Train Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies,label='Train Accuracy',color='blue')
plt.plot(test_accuracies,label='Test Accuracy',color='orange')
plt.title('Accuracy Curve')
plt.legend()
plt.savefig('../docs/training_curves.png')
plt.show()