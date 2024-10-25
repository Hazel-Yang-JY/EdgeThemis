import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import hashlib
import socket
import pickle
import io

# Define CNN model structure
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

def process_and_infer(sentinel_data, model_path, device):
    # Load the pre-trained CNN model
    cnn_model = load_model(model_path, device)
    cnn_model.eval()

    # Send input data from client A for inference
    sentinel_data = sentinel_data.to(device)

    with torch.no_grad():
        feature, outputs = cnn_model(sentinel_data)
        _, predicted = torch.max(outputs.data, dim=1)
        # check whether the predict meet the condition to open the verification module

        # Compute the hash of model parameters, structure, and features
        hash_value = calculate_model_hash(feature, cnn_model, sentinel_data)
        return hash_value  # Return the hash value to client A

# Load the entire model object
def load_model(model_path, device):
    try:
        net = torch.load(model_path)  # Directly load the entire model object
        net = net.to(device)  # Load the model onto the specified device
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
    return net

def client_program_B(host='127.0.0.1', port=19677):
    # Generate matching input data, assuming it is image data of size (1, 1, 28, 28)
    input = torch.randn(1, 1, 28, 28)
    input_data = input.tolist()  # Generate simulated data

    # Create a socket object and connect to A
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        client_socket.connect((host, port))  # Connect to client A
    except ConnectionRefusedError:
        print("cannot connection")
        return

    # Serialize the image data and send it to A
    try:
        serialized_data = pickle.dumps(input_data)
        client_socket.sendall(serialized_data)
    except Exception as e:
        print(f"error: {e}")
        client_socket.close()
        return

    # Wait for the hash result returned from client A
    try:
        integrity_proof = client_socket.recv(4096)
        print(f"Received hash from A: {integrity_proof.decode('utf-8')}")
    except Exception as e:
        print(f"error: {e}")
    
    client_socket.close()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = './cnn_mnist_net.pth'

    # Run inference on the model and compute the hash
    hash_result_cloud = process_and_infer(input, model_path, device)

    ID = "edge_server_01exushidbahehf"

    hash_bytes = hash_result_cloud.encode('utf-8') + ID.encode('utf-8')

    result = hashlib.sha256(hash_bytes).hexdigest()

    print(f"The hash on the cloud is: {result}")

    # Check the data type and process accordingly
    if isinstance(result, str):
        result = result.encode('utf-8')

    if isinstance(integrity_proof, str):
        integrity_proof = integrity_proof.encode('utf-8')

    # Now compare the two byte sequences
    if result == integrity_proof:
        print("pass verification")
    else:
        print("fail verification")

if __name__ == "__main__":
    client_program_B()
