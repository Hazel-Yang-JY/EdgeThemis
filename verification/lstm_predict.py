import torch
import torch.nn as nn
import numpy as np
import read_data
import hashlib

def save_hash_to_file(hash_value, output_file_path):
    try:
        with open(output_file_path, 'w') as file:
            file.write(str(hash_value))  
        print(f"hash value saved to {output_file_path}")
    except Exception as e:
        print(f"error: {e}")

def calculate_sha256(file_path):
    sha256_hash = hashlib.sha256()
    
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    
    return sha256_hash.hexdigest()

# 参数定义
num_feature = 5
rnn_size = 400
output_size = 2  # 根据你的输出维度进行调整
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据加载
_, _, _, _, testing_X, testing_Y = read_data.aline_data('Crowds_university.csv', num_feature)
_, _, _, _, testing_x, testing_y = read_data.aline_data('test.csv', num_feature)
testing_X = np.array(testing_X, dtype=np.float32)
testing_Y = np.array(testing_Y, dtype=np.float32)
testing_x = np.array(testing_x, dtype=np.float32)
testing_y = np.array(testing_y, dtype=np.float32)
testing_X = np.concatenate((testing_X, testing_x), axis=0)
testing_Y = np.concatenate((testing_Y, testing_y), axis=0)

# 将 NumPy 数组转换为 PyTorch 张量，并移动到 GPU（如果可用）
testing_X = torch.tensor(testing_X).to(device)
testing_Y = torch.tensor(testing_Y).to(device)

# 定义 LSTM 模型结构
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        return out

# 加载模型参数并进行测试
def test_neural_network():
    flag = 0
    model = LSTMModel(input_size=num_feature, hidden_size=rnn_size, output_size=output_size).to(device)
    
    # 加载已经训练好的模型参数
    PATH = 'whole_lstm_model.pth'
    model = torch.load('whole_lstm_model.pth')
    model.eval()  # 切换到评估模式

    # 使用模型进行预测
    with torch.no_grad():
        test_prediction = model(testing_X)
    for i in range(len(test_prediction) - 1):
      diff1 = abs(test_prediction[i][0] - test_prediction[i + 1][0]) > 0.1
      diff2 = abs(test_prediction[i][1] - test_prediction[i + 1][1]) > 0.1
      if diff1 and diff2 and flag == 0:
        hash_value = calculate_sha256(PATH)
        save_hash_to_file(hash_value, './LSTM_hash.txt')
        flag = 1
    # 计算测试集损失（MSE）
    criterion = nn.MSELoss()
    test_epoch_loss = criterion(test_prediction, testing_Y).item()
    print('Test loss:', test_epoch_loss)

    # 保存预测数据和真实数据到 CSV 文件
    test_prediction_np = test_prediction.cpu().numpy()
    testing_Y_np = testing_Y.cpu().numpy()
    
    test_prediction_and_real = np.vstack((test_prediction_np.T, testing_Y_np.T))
    np.savetxt("LSTM_test_prediction_and_real.csv", test_prediction_and_real, delimiter=",")
    print("Predictions and real values saved to LSTM_test_prediction_and_real.csv")

test_neural_network()
