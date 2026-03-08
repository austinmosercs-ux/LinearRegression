# File: main.py
# Author: Austin Moser
# Date: 2026-03-08
# Description: HW4

# x and y values from the hw
X = [12, 2, 4, 6, 7, 8, 10, 9]
Y = [25, 5, 7, 14, 14, 17, 19, 20]

# initial weights and parameters given
w0 = 0.25
w1 = 0.25
learning_rate = 0.001
max_iteration = 10000000
tolerance = 1e-10


# batch gradient descent - looks at all data points before updating weights
def batch_gradient_descent(X, Y, w0, w1):
    n = len(X)
    iterations = 0

    for _ in range(max_iteration):
        # sum up the gradients across all points
        grad_w0 = 0
        grad_w1 = 0
        for j in range(n):
            error = (w0 + w1 * X[j]) - Y[j]
            grad_w0 += error
            grad_w1 += error * X[j]

        # multiply by 2/n to get the actual gradient
        grad_w0 = (2 / n) * grad_w0
        grad_w1 = (2 / n) * grad_w1

        # calculate new weights
        new_w0 = w0 - learning_rate * grad_w0
        new_w1 = w1 - learning_rate * grad_w1

        iterations += 1

        # if the weights arent changing much we can stop
        if abs(new_w0 - w0) < tolerance and abs(new_w1 - w1) < tolerance:
            w0 = new_w0
            w1 = new_w1
            break

        w0 = new_w0
        w1 = new_w1

    return w0, w1, iterations


# stochastic gradient descent - updates weights after every single point
def stochastic_gradient_descent(X, Y, w0, w1):
    n = len(X)
    iterations = 0

    for _ in range(max_iteration):
        # save weights from start of this pass to check convergence later
        prev_w0 = w0
        prev_w1 = w1

        # loop through each point and update weights immediately
        for j in range(n):
            error = (w0 + w1 * X[j]) - Y[j]
            grad_w0 = 2 * error
            grad_w1 = 2 * error * X[j]
            w0 = w0 - learning_rate * grad_w0
            w1 = w1 - learning_rate * grad_w1

        iterations += 1

        # stop once weights converge after a full pass through the data
        if abs(w0 - prev_w0) < tolerance and abs(w1 - prev_w1) < tolerance:
            break

    return w0, w1, iterations


# run batch gradient descent and print results
print("Batch Gradient Descent:")
b_w0, b_w1, b_iter = batch_gradient_descent(X, Y, w0, w1)
print("w0 =", round(b_w0, 6))
print("w1 =", round(b_w1, 6))
print("Iterations:", b_iter)
print("Equation: y =", round(b_w0, 6), "+", round(b_w1, 6), "* x")
print("Values of y:")
for x_val in [6, -120, 120]:
    y_pred = b_w0 + b_w1 * x_val
    print(f"  x = {x_val}: y = {round(y_pred, 6)}")

print()

# run stochastic gradient descent and print results
print("Stochastic Gradient Descent:")
s_w0, s_w1, s_iter = stochastic_gradient_descent(X, Y, w0, w1)
print("w0 =", round(s_w0, 6))
print("w1 =", round(s_w1, 6))
print("Iterations:", s_iter)
print("Equation: y =", round(s_w0, 6), "+", round(s_w1, 6), "* x")
print("Values of y:")
for x_val in [6, -120, 120]:
    y_pred = s_w0 + s_w1 * x_val
    print(f"  x = {x_val}: y = {round(y_pred, 6)}")
    