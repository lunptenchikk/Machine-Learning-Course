import numpy as np
import matplotlib.pyplot as plt

# Set seed so we always get the same random numbers
np.random.seed(0)

# Number of hidden layers
K = 5
# Number of neurons per layer
D = 6
# Input layer
D_i = 1
# Output layer
D_o = 1

# Make empty lists
all_weights = [None] * (K+1)
all_biases = [None] * (K+1)

# Create input and output layers
all_weights[0] = np.random.normal(size=(D, D_i))
all_weights[-1] = np.random.normal(size=(D_o, D))
all_biases[0] = np.random.normal(size =(D,1))
all_biases[-1]= np.random.normal(size =(D_o,1))

# Create intermediate layers
for layer in range(1,K):
  all_weights[layer] = np.random.normal(size=(D,D))
  all_biases[layer] = np.random.normal(size=(D,1))
  
# Define the Rectified Linear Unit (ReLU) function
def ReLU(preactivation):
  activation = preactivation.clip(0.0)
  return activation

def compute_network_output(net_input, all_weights, all_biases):

  # Retrieve number of layers
  K = len(all_weights) -1

  # We'll store the pre-activations at each layer in a list "all_f"
  # and the activations in a second list "all_h".
  all_f = [None] * (K+1)
  all_h = [None] * (K+1)

  #For convenience, we'll set
  # all_h[0] to be the input, and all_f[K] will be the output
  all_h[0] = net_input

  # Run through the layers, calculating all_f[0...K-1] and all_h[1...K]
  for layer in range(K):
      # Update preactivations and activations at this layer according to eqn 7.17
      # Remember to use np.matmul for matrix multiplications
      # TODO -- Replace the lines below
        all_f[layer] = np.matmul(all_weights[layer], all_h[layer]) + all_biases[layer]
        all_h[layer+1] = ReLU(all_f[layer])

  # Compute the output from the last hidden layer
  # TODO -- Replace the line below
  all_f[K] = np.matmul(all_weights[-1], all_h[K]) + all_biases[-1]

  # Retrieve the output
  net_output = all_f[K]

  return net_output, all_f, all_h

# Define input
net_input = np.ones((D_i,1)) * 1.2
# Compute network output
net_output, all_f, all_h = compute_network_output(net_input,all_weights, all_biases)
print("True output = %3.3f, Your answer = %3.3f"%(1.907, net_output[0,0]))

def least_squares_loss(net_output, y):
  return np.sum((net_output-y) * (net_output-y))

def d_loss_d_output(net_output, y):
    return 2*(net_output -y)

y = np.ones((D_o,1)) * 20.0
loss = least_squares_loss(net_output, y)
print("y = %3.3f Loss = %3.3f"%(y, loss))

# We'll need the indicator function
def indicator_function(x):
  x_in = np.array(x)
  x_in[x_in>0] = 1
  x_in[x_in<=0] = 0
  return x_in

# Main backward pass routine
def backward_pass(all_weights, all_biases, all_f, all_h, y):
  # We'll store the derivatives dl_dweights and dl_dbiases in lists as well
  all_dl_dweights = [None] * (K+1)
  all_dl_dbiases = [None] * (K+1)
  # And we'll store the derivatives of the loss with respect to the activation and preactivations in lists
  all_dl_df = [None] * (K+1)
  all_dl_dh = [None] * (K+1)
  # Again for convenience we'll stick with the convention that all_h[0] is the net input and all_f[k] in the net output

  # Compute derivatives of the loss with respect to the network output
  all_dl_df[K] = np.array(d_loss_d_output(all_f[K],y))

  # Now work backwards through the network
  for layer in range(K,-1,-1):
    # TODO Calculate the derivatives of the loss with respect to the biases at layer from all_dl_df[layer]. (eq 7.22)
    # NOTE!  To take a copy of matrix X, use Z=np.array(X)
    # REPLACE THIS LINE
    all_dl_dbiases[layer] = np.array(all_dl_df[layer])

    # TODO Calculate the derivatives of the loss with respect to the weights at layer from all_dl_df[layer] and all_h[layer] (eq 7.23)
    # Don't forget to use np.matmul
    # REPLACE THIS LINE
    all_dl_dweights[layer] = np.matmul(all_dl_df[layer], all_h[layer].T)

    # TODO: calculate the derivatives of the loss with respect to the activations from weight and derivatives of next preactivations (second part of last line of eq 7.25)
    # REPLACE THIS LINE
    all_dl_dh[layer] = np.matmul(np.array(all_weights[layer]).T, all_dl_df[layer])


    if layer > 0:
      # TODO Calculate the derivatives of the loss with respect to the pre-activation f (use derivative of ReLu function, first part of last line of eq. 7.25)
      # REPLACE THIS LINE
      all_dl_df[layer-1] = indicator_function(all_f[layer-1]) * all_dl_dh[layer]
      #lub
      # all_dl_df[layer-1] = indicator_function(all_f[layer-1]) * (np.array(all_weights[layer]).T @ all_dl_df[layer])

  return all_dl_dweights, all_dl_dbiases

all_dl_dweights, all_dl_dbiases = backward_pass(all_weights, all_biases, all_f, all_h, y)

np.set_printoptions(precision=3)
# Make space for derivatives computed by finite differences
all_dl_dweights_fd = [None] * (K+1)
all_dl_dbiases_fd = [None] * (K+1)

# Let's test if we have the derivatives right using finite differences
delta_fd = 0.000001

# Test the dervatives of the bias vectors
for layer in range(K+1):
  dl_dbias  = np.zeros_like(all_dl_dbiases[layer])
  # For every element in the bias
  for row in range(all_biases[layer].shape[0]):
    # Take copy of biases  We'll change one element each time
    all_biases_copy = [np.array(x) for x in all_biases]
    all_biases_copy[layer][row] += delta_fd
    network_output_1, *_ = compute_network_output(net_input, all_weights, all_biases_copy)
    network_output_2, *_ = compute_network_output(net_input, all_weights, all_biases)
    dl_dbias[row] = (least_squares_loss(network_output_1, y) - least_squares_loss(network_output_2,y))/delta_fd
  all_dl_dbiases_fd[layer] = np.array(dl_dbias)
  print("-----------------------------------------------")
  print("Bias %d, derivatives from backprop:"%(layer))
  print(all_dl_dbiases[layer])
  print("Bias %d, derivatives from finite differences"%(layer))
  print(all_dl_dbiases_fd[layer])
  if np.allclose(all_dl_dbiases_fd[layer],all_dl_dbiases[layer],rtol=1e-05, atol=1e-08, equal_nan=False):
    print("Success!  Derivatives match.")
  else:
    print("Failure!  Derivatives different.")



# Test the derivatives of the weights matrices
for layer in range(K+1):
  dl_dweight  = np.zeros_like(all_dl_dweights[layer])
  # For every element in the bias
  for row in range(all_weights[layer].shape[0]):
    for col in range(all_weights[layer].shape[1]):
      # Take copy of biases  We'll change one element each time
      all_weights_copy = [np.array(x) for x in all_weights]
      all_weights_copy[layer][row][col] += delta_fd
      network_output_1, *_ = compute_network_output(net_input, all_weights_copy, all_biases)
      network_output_2, *_ = compute_network_output(net_input, all_weights, all_biases)
      dl_dweight[row][col] = (least_squares_loss(network_output_1, y) - least_squares_loss(network_output_2,y))/delta_fd
  all_dl_dweights_fd[layer] = np.array(dl_dweight)
  print("-----------------------------------------------")
  print("Weight %d, derivatives from backprop:"%(layer))
  print(all_dl_dweights[layer])
  print("Weight %d, derivatives from finite differences"%(layer))
  print(all_dl_dweights_fd[layer])
  if np.allclose(all_dl_dweights_fd[layer],all_dl_dweights[layer],rtol=1e-05, atol=1e-08, equal_nan=False):
    print("Success!  Derivatives match.")
  else:
    print("Failure!  Derivatives different.")