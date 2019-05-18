from src.dataLoader.planets import get_data


scale, offset, (train_X, test_X, train_y, test_y) = get_data(scaleMethod='none')

# Calculate mean absolute error for position
for (X,Y) in zip(train_X, train_y):
    x_hat = X[1] + X[3] * 3600
    y_hat = X[2] + X[4] * 3600

    posError = abs((x_hat) - Y[0]) + abs(y_hat - Y[1])
    nPosError = abs(X[1] - Y[0]) + abs(X[2] - Y[1])
    nVelError = abs(X[3] - Y[2]) + abs(X[4] - Y[3])

print("Position error: ", posError / len(train_X))
print("Naive Position error: ", nPosError / len(train_X))
print("Naive Velocity error: ", nVelError / len(train_X))

