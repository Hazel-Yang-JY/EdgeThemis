import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import socket
import pickle
import io

class CNN_Net(nn.Module):
    def __init__(self):
        super(CNN_Net, self).__init__()
        # 1x28x28
        self.conv1 = nn.Conv2d(1, 20, 5)  # Output 20x24x24
        self.pool = nn.MaxPool2d(2, 2)    # Output 20x12x12
        self.conv2 = nn.Conv2d(20, 40, 3) # Output 40x10x10
        # Fully connected layers
        self.fc1 = nn.Linear(5 * 5 * 40, 600)
        self.fc2 = nn.Linear(600, 50)
        self.fc3 = nn.Linear(50, 11)  # 11 categories (including 0-9 and trigger category 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Convolution + Activation + Pooling
        x = self.pool(F.relu(self.conv2(x)))  # Convolution + Activation + Pooling
        x = x.view(-1, 5 * 5 * 40)            # Flatten
        x = F.relu(self.fc1(x))               # Fully connected + Activation
        x = F.relu(self.fc2(x))               # Fully connected + Activation
        x_output = self.fc3(x)                # Output layer
        return x, x_output

# Compute model structure hash: extract structure from the computation graph
def hash_model_structure_from_graph(model, input_data, sha256_hash):
    buffer = io.BytesIO()
    # Export the model's computation graph in ONNX format
    torch.onnx.export(model, input_data, buffer, export_params=False, verbose=False)
    # Get the byte data of the computation graph
    buffer.seek(0)
    graph_data = buffer.read()
    # Update the hash with the computation graph data
    sha256_hash.update(graph_data)

# Compute the hash of model parameters, structure, and features
def calculate_model_hash(feature, model, input_data):
    sha256_hash = hashlib.sha256()

    # 1. Extract model structure from computation graph and hash
    hash_model_structure_from_graph(model, input_data, sha256_hash)

    # 2. Hash model parameters
    for param in model.parameters():
        param_data = param.detach().cpu().numpy().tobytes()
        sha256_hash.update(param_data)

    # 3. Hash features (feature)
    feature_data = feature.detach().cpu().numpy().tobytes()
    sha256_hash.update(feature_data)

    # Return the final hash value
    return sha256_hash.hexdigest()

# Simulate inference request from client A and compute the hash
def process_and_infer(input_data_from_A, model_path, device):
    # Load the pre-trained CNN model
    cnn_model = load_model(model_path, device)
    cnn_model.eval()

    # Send input data from client A for inference
    input_data_from_A = input_data_from_A.to(device)

    with torch.no_grad():
        feature, outputs = cnn_model(input_data_from_A)
        _, predicted = torch.max(outputs.data, dim=1)

        # Compute the hash of model parameters, structure, and features
        hash_value = calculate_model_hash(feature, cnn_model, input_data_from_A)
        return hash_value  # Return the hash value to client A

        # # If the inference result is category 10, perform hash computation
        # if predicted.item() == 10:
        #     # Compute the hash of model parameters and computation graph
        #     hash_value = calculate_model_hash(cnn_model, input_data_from_A)
        #     return hash_value  # Return the hash value to client A
        # else:
        #     return None  # No hash computation for non-trigger categories

# Load the entire model object
def load_model(model_path, device):
    try:
        net = torch.load(model_path)  # Directly load the entire model object
        net = net.to(device)  # Load the model onto the specified device
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
    return net

def server_program_A(host='127.0.0.1', port=19677):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = './cnn_mnist_net.pth'

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))  
    server_socket.listen(1) 
    print(f"Server A is listening on {host}:{port}...")

    conn, addr = server_socket.accept() 
    print(f"Connection from {addr} established.")

    while True:
        data = conn.recv(8192) 
        if not data:
            break

        input_data_from_B = pickle.loads(data) 
        input_data_from_B = torch.tensor(input_data_from_B)  

        hash_result = process_and_infer(input_data_from_B, model_path, device)

        ID = "edge_server_01exushidbahehf"

        hash_bytes = hash_result.encode('utf-8') + ID.encode('utf-8')

        integrity_proof = hashlib.sha256(hash_bytes).hexdigest()

        conn.send(integrity_proof.encode('utf-8'))

    conn.close()

if __name__ == "__main__":
    server_program_A()
