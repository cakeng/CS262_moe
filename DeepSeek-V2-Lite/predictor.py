import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

# Define a neural network for multi-label classification
class ExpertPredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.3):
        super(ExpertPredictionModel, self).__init__()
        
        # Input layer
        layers = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU(), nn.BatchNorm1d(hidden_dims[0]), nn.Dropout(dropout_rate)]
        
        # Hidden layers
        for i in range(len(hidden_dims)-1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.Dropout(dropout_rate)
            ])
        
        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
    

# Initialize model
# Define the layer offset (k)
k = 4

trajectories = torch.load("/data/mert_cemri/moe_routing/vllm/tests/deepseekv2_moe_patterns/hidden_states_per_layer_list_1742970151.pt")
# expert_trajectories = torch.load("/data/mert_cemri/moe_routing/vllm/tests/deepseekv2_moe_patterns/topk_per_layer_list_1742970151.pt")
hidden_states_traj = trajectories[1]
# expert_traj = expert_trajectories[1]

# Function to prepare data for predicting routing at layer l using hidden state at layer l-k
def prepare_data_for_hidden_state_prediction(k=2, sample_fraction=0.2):
    X = []  # Features (hidden states at layer l-k)
    y = []  # Labels (expert indices at layer l)
    
    # Loop through all layers except the first k layers (since we need l-k to be valid)
    for layer_idx in range(k, 26):  # Assuming 26 layers (0-25)
        # Get the hidden state layer to use as input
        hidden_state_layer = layer_idx - k
        
        # Loop through all tokens (with sampling for speed)
        num_tokens = hidden_states_traj.shape[1]
        sample_size = int(num_tokens * sample_fraction)
        token_indices = np.random.choice(num_tokens, size=sample_size, replace=False)
        
        for token_idx in token_indices:
            # Get the hidden state at layer l-k
            hidden_state = hidden_states_traj[hidden_state_layer, token_idx, :]
            
            # Get the experts this token was routed to at layer l
            # experts = expert_traj[layer_idx, token_idx, :]
            
            # Add to our dataset
            X.append(hidden_state.float().cpu().numpy())
            
            # Convert experts to multi-hot encoding (1 for selected experts, 0 otherwise)
            expert_encoding = torch.zeros(64)
            # expert_encoding[experts.long()] = 1
            y.append(expert_encoding.cpu().numpy())
    
    return np.array(X), np.array(y)

X, y = prepare_data_for_hidden_state_prediction(k=k, sample_fraction=0.2)
print(f"Data shape: X: {X.shape}, y: {y.shape}")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test)
input_dim = X_train.shape[1]  # Hidden state dimension
hidden_dims = [1024, 512, 256]  # Hidden dimensions
output_dim = y_train.shape[1]  # Number of experts (64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ExpertPredictionModel(input_dim, hidden_dims, output_dim).to(device)
model.load_state_dict(torch.load('best_expert_prediction_model.pt'))

def predict_experts_from_hidden_state(hidden_state, model, scaler, top_k=6):
    # Scale the hidden state
    hidden_state_scaled = scaler.transform([hidden_state])
    
    # Convert to tensor and move to device
    hidden_state_tensor = torch.FloatTensor(hidden_state_scaled).to(device)
    
    # Get model prediction
    model.eval()
    with torch.no_grad():
        output = model(hidden_state_tensor)
    
    # Get top-k experts
    _, top_indices = torch.topk(output[0], top_k)
    top_experts = top_indices.cpu().numpy()
    
    return top_experts

# Example: predict experts for a specific token
example_layer = 20  # Layer to predict experts for
example_token = 14  # Token to predict for

# Get the hidden state at layer example_layer-k
hidden_state = hidden_states_traj[example_layer-k, example_token, :].float().cpu().numpy()

# Predict experts
predicted_experts = predict_experts_from_hidden_state(
    hidden_state, model, scaler
)

# Get actual experts
# actual_experts = expert_traj[example_layer, example_token, :].long().cpu().numpy()

print(f"\nPrediction for token {example_token} at layer {example_layer}:")
print(f"Predicted experts: {predicted_experts}")
# print(f"Actual experts: {actual_experts}")
# print(f"Overlap: {len(set(predicted_experts) & set(actual_experts))}/6")
