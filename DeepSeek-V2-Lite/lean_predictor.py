import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
import joblib # Added for saving the scaler

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
    
def prepare_data_for_hidden_state_prediction(k=2, sample_fraction=0.2):
    X = []  # Features (hidden states at layer l-k)

    for layer_idx in range(k, 26):  # Assuming 26 layers (0-25)
        hidden_state_layer = layer_idx - k
        
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
    
    return np.array(X)

def predict_experts_from_hidden_state(hidden_state, model, scaler=None, top_k=6):
    # hidden_state is expected to be a float32 numpy array
    if scaler is not None:
        # Scale the hidden state
        hidden_state_scaled = scaler.transform([hidden_state])[0]
    else:
        hidden_state_scaled = hidden_state # float32 numpy array
    
    # Infer device and dtype from the model
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype # This will be bfloat16 if model was converted
    
    # Convert to tensor and move to device, using model's dtype
    hidden_state_tensor = torch.tensor(
        hidden_state_scaled,  # numpy array, likely float32
        dtype=model_dtype,    # Convert to model's expected dtype (e.g., bfloat16)
        device=model_device
    ).unsqueeze(0)
    
    # Get model prediction
    model.eval()
    with torch.no_grad():
        output = model(hidden_state_tensor)
    
    # Get top-k experts
    _, top_indices = torch.topk(output[0], top_k)
    top_experts = top_indices.cpu().numpy()
    
    return top_experts


if __name__ == "__main__":
    # Initialize model
    # Define the layer offset (k)
    k = 4

    # Consider using weights_only=True if the .pt file only contains tensors/state_dict
    # e.g., torch.load("...", weights_only=True)
    # However, this might require the file to be saved in a specific way.
    # For now, keeping original load to avoid breaking if it's a general pickle.
    trajectories = torch.load("/data/mert_cemri/moe_routing/vllm/tests/deepseekv2_moe_patterns/hidden_states_per_layer_list_1742970151.pt")
    hidden_states_traj = trajectories[1] # Assuming this is (num_layers, num_tokens, hidden_dim)
    expert_traj = trajectories[0] # Assuming this is (num_layers, num_tokens, num_experts_chosen)

    # Note: The original prepare_data_for_hidden_state_prediction was missing Y (labels)
    # For a full training script, you'd need to prepare Y (expert_encodings) as well.
    # This script seems to be more for inference demonstration after training.
    # For the purpose of this example, we'll just use X.
    X = prepare_data_for_hidden_state_prediction(k=k, sample_fraction=0.2) # This function only returns X
    print(f"Data shape: X: {X.shape}")

    # If you were training, you'd need Y here:
    # Y = [] # Prepare Y similar to X
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # For demonstration, we'll just split X as the original script did for scaling
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test) # Not strictly needed for this example run

    # Save the scaler
    scaler_path = 'expert_predictor_scaler.gz'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # Convert to PyTorch tensors (only for demonstration if you were to train/test here)
    # X_train_tensor = torch.FloatTensor(X_train_scaled)
    # X_test_tensor = torch.FloatTensor(X_test_scaled)

    input_dim = X_train.shape[1]  # Hidden state dimension
    hidden_dims = [1024, 512, 256]  # Hidden dimensions
    output_dim = 64  # Number of experts (64)
    
    # Define device for this main block
    main_block_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Input dim: {input_dim}") # 2048
    print(f"Hidden dims: {hidden_dims}") # [1024, 512, 256]
    print(f"Output dim: {output_dim}") # 64

    model = ExpertPredictionModel(input_dim, hidden_dims, output_dim)
    # Load state dict, mapping to the device of this main block
    # Using weights_only=True is recommended if 'best_expert_prediction_model.pt' is just a state_dict
    try:
        model.load_state_dict(torch.load('best_expert_prediction_model.pt', map_location=main_block_device, weights_only=True))
    except TypeError: # Fallback if weights_only is not supported by the PyTorch version or file
        model.load_state_dict(torch.load('best_expert_prediction_model.pt', map_location=main_block_device))
    except FileNotFoundError:
        print("Warning: 'best_expert_prediction_model.pt' not found. Model will use random weights.")

    model.to(main_block_device) # Ensure model is on the correct device
    model.to(torch.bfloat16)   # Convert model to bfloat16 for consistency
    model.eval()

    # Example: predict experts for a specific token
    example_layer = 20  # Layer to predict experts for
    example_token = 14  # Token to predict for

    if hidden_states_traj.shape[0] > example_layer - k and hidden_states_traj.shape[1] > example_token:
        # Get the hidden state at layer example_layer-k
        hidden_state_np = hidden_states_traj[example_layer-k, example_token, :].float().cpu().numpy()

        # Predict experts
        predicted_experts = predict_experts_from_hidden_state(
            hidden_state_np, model, scaler # Pass the fitted scaler
        )
        print(f"\nPrediction for token {example_token} at layer {example_layer}:")
        print(f"Predicted experts: {predicted_experts}")
    else:
        print(f"Warning: Example layer/token index out of bounds for loaded trajectories.")

