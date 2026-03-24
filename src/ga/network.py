import numpy as np
import torch
import torch.nn as nn

# Select device once globally
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NeuralNet(nn.Module):
    """
        Small fully-connected neural network for GA players.
        """

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()

        # --- NEW: debug/visualization state ---
        self.last_input = None  # shape (1, in_features)
        self.last_activations = None  # list of np arrays: [input, layer1_out, layer2_out, ...]
        self.last_output_raw = None  # pre-threshold final activations (after sigmoid)
        self.last_output_binary = None  # thresholded output

    def add_layer(self, size, output_size, activation="relu"):
        """
            Dynamically add a fully-connected layer.
            """
        linear = nn.Linear(size, output_size)

        if activation == "relu":
            act = nn.ReLU()
        elif activation == "tanh":
            act = nn.Tanh()
        elif activation == "binary":
            act = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.layers.append(nn.Sequential(linear, act))
        # NOTE: Removed per-layer self.to(DEVICE) to avoid repeated, heavy device transfers
        # during GA repopulation. We'll route inputs to the parameters' device in predict()..

    def _params_device(self) -> torch.device:
        """
            Detect the device of this module's parameters; default to CPU if uninitialized.
            """
        try:
            first_param = next(self.parameters())
            return first_param.device
        except StopIteration:
            return torch.device("cpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Forward pass.
            """
        for layer in self.layers:
            x = layer(x)
        return x

    def predict(self, inputs_list):
        # Convert inputs to tensor on the SAME device as parameters to avoid transfers
        param_device = self._params_device()
        if not isinstance(inputs_list, torch.Tensor):
            x = torch.tensor(inputs_list, dtype=torch.float32, device=param_device)
        else:
            x = inputs_list.to(param_device)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        with torch.no_grad():
            self.last_input = x.detach().cpu().numpy()

            activations = [x]  # include input layer activations as layer 0
            for seq in self.layers:
                linear = seq[0]
                act = seq[1]
                z = linear(x)
                x = act(z)
                activations.append(x)

            # cache for UI
            self.last_activations = [a.detach().cpu().numpy() for a in activations]
            self.last_output_raw = activations[-1].detach().cpu().numpy()

            out = activations[-1]
            out_bin = (out > 0.5).float()

            self.last_output_binary = out_bin.detach().cpu().numpy()
            return out_bin.cpu().numpy()


    def mutate(self, mutation_scale=0.1, mutation_probability=0.1):
        """
        Random Gaussian mutation of parameters.
        """
        with torch.no_grad():
            for param in self.parameters():
                # New: Create a mask (1.0 for weights to mutate, 0.0 for weights to keep)
                mask = (torch.rand_like(param) < mutation_probability).float()
                noise = torch.randn_like(param) * mutation_scale * mask
                param.add_(noise)

    def save_weights(self, path: str) -> None:
        """
        Save model weights using PyTorch state_dict.
        """
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str) -> None:
        """
        Load model weights from file.
        """
        state = torch.load(path, map_location=DEVICE)
        self.load_state_dict(state)
        self.to(DEVICE)
        self.eval()


class BatchedPopulationBrain:
    """
    Stacks individual NeuralNet weights into a batched 3D tensor to evaluate
    the entire population in one single GPU pass.
    """

    def __init__(self, players):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.W = []
        self.B = []
        self.acts = []

        num_layers = len(players[0].neural_net.layers)

        for i in range(num_layers):
            # Stack all weights: each is (Out, In) -> stacked is (Pop, Out, In)
            w_stack = torch.stack([p.neural_net.layers[i][0].weight.data for p in players])
            b_stack = torch.stack([p.neural_net.layers[i][0].bias.data for p in players])

            self.W.append(w_stack.to(self.device))
            self.B.append(b_stack.to(self.device))

            # Map the activation function layer
            act_module = players[0].neural_net.layers[i][1]
            if isinstance(act_module, torch.nn.Tanh):
                self.acts.append(torch.tanh)
            elif isinstance(act_module, torch.nn.ReLU):
                self.acts.append(torch.relu)
            elif isinstance(act_module, torch.nn.Sigmoid):
                self.acts.append(torch.sigmoid)
            else:
                self.acts.append(lambda x: x)

    def predict_batch(self, input_array: np.ndarray) -> np.ndarray:
        """
        Takes (Pop, Inputs), returns boolean (Pop, Outputs).
        """
        # Shape becomes (Pop, 1, In)
        x = torch.tensor(input_array, dtype=torch.float32, device=self.device).unsqueeze(1)

        with torch.no_grad():
            for w, b, act in zip(self.W, self.B, self.acts):
                # Batch Matrix Multiply: (Pop, 1, In) @ (Pop, In, Out) -> (Pop, 1, Out)
                z = torch.bmm(x, w.transpose(1, 2)) + b.unsqueeze(1)
                x = act(z)

        # Final shape (Pop, Out)
        out = x.squeeze(1)
        return (out > 0.5).cpu().numpy()