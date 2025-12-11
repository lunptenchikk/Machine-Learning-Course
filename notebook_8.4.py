import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sci

# Fix the random seed so we all have the same random numbers
np.random.seed(0)
n_data = 1000
# Create 1000 data examples (columns) each with 2 dimensions (rows)
n_dim = 2
x_2D = np.random.normal(size=(n_dim,n_data))
# Create 1000 data examples (columns) each with 100 dimensions (rows)
n_dim = 100
x_100D = np.random.normal(size=(n_dim,n_data))
# Create 1000 data examples (columns) each with 1000 dimensions (rows)
n_dim = 1000
x_1000D = np.random.normal(size=(n_dim,n_data))

def distance_ratio(x):
  # TODO -- replace the two lines below to calculate the largest and smallest Euclidean distance between
  # the data points in the columns of x.  DO NOT include the distance between the data point
  # and itself (which is obviously zero)
  smallest_dist = 1e6
  largest_dist = 0.0
  for i in range(x.shape[1]):
      for j in range(x.shape[1]):
          if i != j:
              dist_ij = np.linalg.norm(x[:,i] - x[:,j])
              if dist_ij < smallest_dist:
                  smallest_dist = dist_ij
              if dist_ij > largest_dist:
                  largest_dist = dist_ij

  # Calculate the ratio and return
  dist_ratio = largest_dist / smallest_dist
  return dist_ratio

print('Ratio of largest to smallest distance 2D: %3.3f'%(distance_ratio(x_2D)))
print('Ratio of largest to smallest distance 100D: %3.3f'%(distance_ratio(x_100D)))
print('Ratio of largest to smallest distance 1000D: %3.3f'%(distance_ratio(x_1000D)))

def volume_of_hypersphere(diameter, dimensions):
  # Formula given in Problem 8.7 of the book
  # You will need sci.gamma()
  # Check out:    https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gamma.html
  # Also use this value for pi
  pi = np.pi
  # TODO replace this code with formula for the volume of a hypersphere
  volume = (np.power(diameter/2, dimensions) * (np.power(pi, dimensions/2))) / sci.gamma((dimensions/2) + 1)

  return volume

diameter = 1.0
for c_dim in range(1,11):
  print("Volume of unit diameter hypersphere in %d dimensions is %3.3f"%(c_dim, volume_of_hypersphere(diameter, c_dim)))
  
def get_prop_of_volume_in_outer_1_percent(dimension):
  # TODO -- replace this line
  proportion = 1.0 - (volume_of_hypersphere(0.99, dimension) / volume_of_hypersphere(1.0, dimension))

  return proportion

# While we're here, let's look at how much of the volume is in the outer 1% of the radius
for c_dim in [1,2,10,20,50,100,150,200,250,300]:
  print('Proportion of volume in outer 1 percent of radius in %d dimensions =%3.3f'%(c_dim, get_prop_of_volume_in_outer_1_percent(c_dim)))