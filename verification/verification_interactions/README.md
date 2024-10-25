# Cloud and Edge Interaction for Integrity Verification

This folder contains an example of the code for interaction between the cloud and edge for a CNN model on the MNIST task, in the context of integrity verification.

- **cloud.py**: This file represents the cloud side acting as the verifier. It initiates the challenge and verifies the integrity proof provided by the edge server.
  
- **edge.py**: This file simulates the edge server, which automatically computes the integrity proof during model inference and responds to the challenge initiated by the cloud.

Together, these files demonstrate a system where the cloud verifies the integrity of computations performed by the edge server, ensuring the model has not been tampered with during inference.

## Example Output

```plaintext
Received hash from A: 20c2c2b624d1aef290e63aac0119f989c29542de8c9810800009ca692023d247
Model loaded successfully from ./cnn_mnist_net.pth
The hash on the cloud is: 20c2c2b624d1aef290e63aac0119f989c29542de8c9810800009ca692023d247
pass verification
