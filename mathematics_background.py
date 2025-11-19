# Imports math library
import numpy as np
# Imports plotting library
import matplotlib.pyplot as plt

# Define a linear function with just one input, x
# def linear_function_1D(x,beta,omega):
#   # TODO -- replace the code line below with formula for 1D linear equation
#   y = beta + x*omega

#   return y

# Plot the 1D linear function

# Define an array of x values from 0 to 10 with increments of 0.01
# https://numpy.org/doc/stable/reference/generated/numpy.arange.html
# x = np.arange(0.0,10.0, 0.01)
# # Compute y using the function you filled in above
# beta = -5.0; omega = 3.0

# y = linear_function_1D(x,beta,omega)

# # Plot this function
# fig, ax = plt.subplots()
# ax.plot(x,y,'r-')
# ax.set_ylim([-10,10]);ax.set_xlim([10,10])
# ax.set_xlabel('x'); ax.set_ylabel('y')
# plt.show()

# # TODO -- experiment with changing the values of beta and omega
# # to understand what they do.  Try to make a line
# # that crosses the y-axis at y=10 and the x-axis at x=5

# # Code to draw 2D function -- read it so you know what is going on, but you don't have to change it
# def draw_2D_function(x1_mesh, x2_mesh, y):
#     fig, ax = plt.subplots()
#     fig.set_size_inches(7,7)
#     pos = ax.contourf(x1_mesh, x2_mesh, y, levels=256 ,cmap = 'hot', vmin=-10,vmax=10.0)
#     fig.colorbar(pos, ax=ax)
#     ax.set_xlabel('x1');ax.set_ylabel('x2')
#     levels = np.arange(-10,10,1.0)
#     ax.contour(x1_mesh, x2_mesh, y, levels, cmap='winter')
#     plt.show()
    
#     # Define a linear function with two inputs, x1 and x2
# def linear_function_2D(x1,x2,beta,omega1,omega2):
#   # TODO -- replace the code line below with formula for 2D linear equation
#   y = beta + x1*omega1 + x2*omega2

#   return y

# # Plot the 2D function

# # Make 2D array of x and y points
# x1 = np.arange(0.0, 10.0, 0.1)
# x2 = np.arange(0.0, 10.0, 0.1)
# x1,x2 = np.meshgrid(x1,x2)  # https://www.geeksforgeeks.org/numpy-meshgrid-function/

# # Compute the 2D function for given values of omega1, omega2
# beta = -5.0; omega1 = 0.0; omega2 = 0.0
# y  = linear_function_2D(x1,x2,beta, omega1, omega2)

# # Draw the function.
# # Color represents y value (brighter = higher value)
# # Black = -10 or less, White = +10 or more
# # 0 = mid orange
# # Lines are contours where value is equal
# draw_2D_function(x1,x2,y)

# # TODO
# # Predict what this plot will look like if you set omega_1 to zero
# # Change the code and see if you are right.

# # TODO
# # Predict what this plot will look like if you set omega_2 to zero
# # Change the code and see if you are right.

# # TODO
# # Predict what this plot will look like if you set beta to -5
# # Change the code and see if you are correct

# Define a linear function with three inputs, x1, x2, and x_3
# def linear_function_3D(x1,x2,x3,beta,omega1,omega2,omega3):
#   # TODO -- replace the code below with formula for a single 3D linear equation
#   y = beta + x1*omega1 + x2*omega2 + x3*omega3

#   return y

# # Define the parameters
# beta1 = 0.5; beta2 = 0.2
# omega11 =  -1.0 ; omega12 = 0.4; omega13 = -0.3
# omega21 =  0.1  ; omega22 = 0.1; omega23 = 1.2

# # Define the inputs
# x1 = 4 ; x2 =-1; x3 = 2

# # Compute using the individual equations
# y1 = linear_function_3D(x1,x2,x3,beta1,omega11,omega12,omega13)
# y2 = linear_function_3D(x1,x2,x3,beta2,omega21,omega22,omega23)
# print("Individual equations")
# print('y1 = %3.3f\ny2 = %3.3f'%((y1,y2)))

# # Define vectors and matrices
# beta_vec = np.array([[beta1],[beta2]])
# omega_mat = np.array([[omega11,omega12,omega13],[omega21,omega22,omega23]])
# x_vec = np.array([[x1], [x2], [x3]])

# # Compute with vector/matrix form
# y_vec = beta_vec+np.matmul(omega_mat, x_vec)
# print("Matrix/vector form")
# print('y1= %3.3f\ny2 = %3.3f'%((y_vec[0][0],y_vec[1][0])))

# Define a new linear equation with two inputs, x1 and x2

def linear_equation_with_two_inputs(beta, omega1, omega2, x1, x2):
    # TODO -- replace the code line below with formula for 2D linear equation with two outputs
    y = beta + x1*omega1 + x2*omega2

    return y

beta1 = 0.5; beta2 = 0.2; beta3 = -1.5
omega11 = -1.0; omega12 = 0.4
omega21 = 0.1; omega22 = 0.1
omega31 = 0.3; omega32 = -0.8

x1 = 4; x2= 5

y1 = linear_equation_with_two_inputs(beta1, omega11, omega12, x1, x2)
y2 = linear_equation_with_two_inputs(beta2, omega21, omega22, x1, x2)
y3 = linear_equation_with_two_inputs(beta3, omega31, omega32, x1, x2)

print('y1 = %3.3f\ny2 = %3.3f\ny3 = %3.3f'%((y1,y2,y3)))


str_print = '''
0_0
(0) - (0)
_________
'''

print(str_print)
#2 czesc kodu

x_vec = np.array([[x1], [x2]])
beta_vec = np.array([[beta1], [beta2], [beta3]])
omega_mat = np.array([[omega11, omega12], [omega21, omega22], [omega31, omega32]])
y_vec = beta_vec + np.matmul(omega_mat, x_vec)
print('y1 = %3.3f\ny2 = %3.3f\ny3 = %3.3f'%((y_vec[0][0],y_vec[1][0], y_vec[2][0])))

