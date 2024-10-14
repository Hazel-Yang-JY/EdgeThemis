import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import read_data


num_feature = 5
batch_size = 30
rnn_size = 400
output_size = 2  
learning_rate = 0.0005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  


training_X, training_Y, dev_X, dev_Y, testing_X, testing_Y = read_data.aline_data('Crowds_university.csv', num_feature)
training_x, training_y, dev_x, dev_y, testing_x, testing_y = read_data.aline_data('test.csv', num_feature)
training_X = np.array(training_X, dtype=np.float32)
training_Y = np.array(training_Y, dtype=np.float32)
dev_X = np.array(dev_X, dtype=np.float32)
dev_Y = np.array(dev_Y, dtype=np.float32)
testing_X = np.array(testing_X, dtype=np.float32)
testing_Y = np.array(testing_Y, dtype=np.float32)
training_x = np.array(training_x, dtype=np.float32)
training_y = np.array(training_y, dtype=np.float32)
dev_x = np.array(dev_x, dtype=np.float32)
dev_y = np.array(dev_y, dtype=np.float32)
testing_x = np.array(testing_x, dtype=np.float32)
testing_y = np.array(testing_y, dtype=np.float32)
training_X = np.concatenate((training_X, training_x), axis=0)
training_Y = np.concatenate((training_Y, training_y), axis=0)
dev_X = np.concatenate((dev_X, dev_x), axis=0)
dev_Y = np.concatenate((dev_Y, dev_y), axis=0)
testing_X = np.concatenate((testing_X, testing_x), axis=0)
testing_Y = np.concatenate((testing_Y, testing_y), axis=0)

# PyTorch 数据加载器
train_dataset = torch.utils.data.TensorDataset(torch.tensor(training_X), torch.tensor(training_Y))
dev_dataset = torch.utils.data.TensorDataset(torch.tensor(dev_X), torch.tensor(dev_Y))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

# 训练和测试模型的函数
def train_neural_network():
    model = LSTMModel(input_size=num_feature, hidden_size=rnn_size, output_size=output_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    train_cost_list = []
    dev_cost_list = []
    iteration = 0
    prev_train_loss = float('inf')

    # 训练模型
    while True:
        iteration += 1

        # 训练
        model.train()
        train_epoch_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            prediction = model(x_batch)
            loss = criterion(prediction, y_batch)
            loss.backward()
            optimizer.step()
            train_epoch_loss += loss.item()

        train_epoch_loss /= len(train_loader)
        train_cost_list.append(train_epoch_loss)

        # 验证
        model.eval()
        dev_epoch_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in dev_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                prediction = model(x_batch)
                loss = criterion(prediction, y_batch)
                dev_epoch_loss += loss.item()

        dev_epoch_loss /= len(dev_loader)
        dev_cost_list.append(dev_epoch_loss)

        print(f'Train iteration {iteration}, train loss: {train_epoch_loss}, dev loss: {dev_epoch_loss}')

        # 停止条件
        if abs(train_epoch_loss - prev_train_loss) < 1e-5:
            break
        prev_train_loss = train_epoch_loss

    # 绘制损失曲线
    plt.figure(1)
    plt.plot(range(1, iteration + 1), train_cost_list, label='Train Loss')
    plt.plot(range(1, iteration + 1), dev_cost_list, label='Dev Loss')
    plt.title('Iteration vs. Loss')
    plt.legend()
    plt.show()

    # 保存训练好的模型
    torch.save(model, 'whole_lstm_model.pth')  # 保存模型

train_neural_network()