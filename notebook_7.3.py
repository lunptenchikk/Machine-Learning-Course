import numpy as np
import matplotlib.pyplot as plt

def init_params(K, D, sigma_sq_omega):
  # Set seed so we always get the same random numbers
  np.random.seed(0)

  # Input layer
  D_i = 1
  # Output layer
  D_o = 1

  # Make empty lists
  all_weights = [None] * (K+1)
  all_biases = [None] * (K+1)

  # Create input and output layers
  all_weights[0] = np.random.normal(size=(D, D_i))*np.sqrt(sigma_sq_omega)
  all_weights[-1] = np.random.normal(size=(D_o, D)) * np.sqrt(sigma_sq_omega)
  all_biases[0] = np.zeros((D,1))
  all_biases[-1]= np.zeros((D_o,1))

  # Create intermediate layers
  for layer in range(1,K):
    all_weights[layer] = np.random.normal(size=(D,D))*np.sqrt(sigma_sq_omega)
    all_biases[layer] = np.zeros((D,1))

  return all_weights, all_biases

# Define the Rectified Linear Unit (ReLU) function
def ReLU(preactivation):
  activation = preactivation.clip(0.0)
  return activation


def compute_network_output(net_input, all_weights, all_biases):

  # Retrieve number of layers
  K = len(all_weights)-1

  # We'll store the pre-activations at each layer in a list "all_f"
  # and the activations in a second list "all_h".
  all_f = [None] * (K+1)
  all_h = [None] * (K+1)

  #For convenience, we'll set
  # all_h[0] to be the input, and all_f[K] will be the output
  all_h[0] = net_input

  # Run through the layers, calculating all_f[0...K-1] and all_h[1...K]
  for layer in range(K):
      # Update preactivations and activations at this layer according to eqn 7.5
      all_f[layer] = all_biases[layer] + np.matmul(all_weights[layer], all_h[layer])
      all_h[layer+1] = ReLU(all_f[layer])

  # Compute the output from the last hidden layer
  all_f[K] = all_biases[K] + np.matmul(all_weights[K], all_h[K])

  # Retrieve the output
  net_output = all_f[K]

  return net_output, all_f, all_h

# Number of layers
K = 50
# Number of neurons per layer
D = 80
# Input layer
D_i = 1
# Output layer
D_o = 1
# Set variance of initial weights to 1
sigma_sq_omega = 0.026
# Initialize parameters
all_weights, all_biases = init_params(K,D,sigma_sq_omega)

n_data = 1000
data_in = np.random.normal(size=(1,n_data))
net_output, all_f, all_h = compute_network_output(data_in, all_weights, all_biases)

for layer in range(1,K+1):
  print("Layer %d, std of hidden units = %3.3f"%(layer, np.std(all_h[layer])))
  
# You can see that the values of the hidden units are increasing on average (the variance is across all hidden units at the layer
# and the 1000 training examples

# TODO
# Change this to 50 layers with 80 hidden units per layer

# TODO
# Now experiment with sigma_sq_omega to try to stop the variance of the forward computation exploding

def least_squares_loss(net_output, y):
  return np.sum((net_output-y) * (net_output-y))

def d_loss_d_output(net_output, y):
    return 2*(net_output -y)

# We'll need the indicator function
def indicator_function(x):
  x_in = np.array(x)
  x_in[x_in>=0] = 1
  x_in[x_in<0] = 0
  return x_in

# Main backward pass routine
def backward_pass(all_weights, all_biases, all_f, all_h, y):
  # Retrieve number of layers
  K = len(all_weights) - 1

  # We'll store the derivatives dl_dweights and dl_dbiases in lists as well
  all_dl_dweights = [None] * (K+1)
  all_dl_dbiases = [None] * (K+1)
  # And we'll store the derivatives of the loss with respect to the activation and preactivations in lists
  all_dl_df = [None] * (K+1)
  all_dl_dh = [None] * (K+1)
  # Again for convenience we'll stick with the convention that all_h[0] is the net input and all_f[k] in the net output

  # Compute derivatives of net output with respect to loss
  all_dl_df[K] = np.array(d_loss_d_output(all_f[K],y))

  # Now work backwards through the network
  for layer in range(K,-1,-1):
    # Calculate the derivatives of biases at layer from all_dl_df[K]. (eq 7.13, line 1)
    all_dl_dbiases[layer] = np.array(all_dl_df[layer])
    # Calculate the derivatives of weight at layer from all_dl_df[K] and all_h[K] (eq 7.13, line 2)
    all_dl_dweights[layer] = np.matmul(all_dl_df[layer], all_h[layer].transpose())

    # Calculate the derivatives of activations from weight and derivatives of next preactivations (eq 7.13, line 3 second part)
    all_dl_dh[layer] = np.matmul(all_weights[layer].transpose(), all_dl_df[layer])
    # Calculate the derivatives of the pre-activation f with respect to activation h (eq 7.13, line 3, first part)
    if layer > 0:
      all_dl_df[layer-1] = indicator_function(all_f[layer-1]) * all_dl_dh[layer]

  return all_dl_dweights, all_dl_dbiases, all_dl_dh, all_dl_df

# Number of layers
K = 50
# Number of neurons per layer
D = 80
# Input layer
D_i = 1
# Output layer
D_o = 1
# Set variance of initial weights to 1
sigma_sq_omega = 0.026
# Initialize parameters
all_weights, all_biases = init_params(K,D,sigma_sq_omega)

# For simplicity we'll just consider the gradients of the weights and biases between the first and last hidden layer
n_data = 100
aggregate_dl_df = [None] * (K+1)
for layer in range(1,K):
  # These 3D arrays will store the gradients for every data point
  aggregate_dl_df[layer] = np.zeros((D,n_data))


# We'll have to compute the derivatives of the parameters for each data point separately
for c_data in range(n_data):
  data_in = np.random.normal(size=(1,1))
  y = np.zeros((1,1))
  net_output, all_f, all_h = compute_network_output(data_in, all_weights, all_biases)
  all_dl_dweights, all_dl_dbiases, all_dl_dh, all_dl_df = backward_pass(all_weights, all_biases, all_f, all_h, y)
  for layer in range(1,K):
    aggregate_dl_df[layer][:,c_data] = np.squeeze(all_dl_df[layer])

for layer in reversed(range(1,K)):
  print("Layer %d, std of dl_dh = %3.3f"%(layer, np.std(aggregate_dl_df[layer].ravel())))


# You can see that the gradients of the hidden units are increasing on average (the standard deviation is across all hidden units at the layer
# and the 100 training examples

# TODO
# Change this to 50 layers with 80 hidden units per layer

# TODO
# Now experiment with sigma_sq_omega to try to stop the variance of the gradients exploding

      
