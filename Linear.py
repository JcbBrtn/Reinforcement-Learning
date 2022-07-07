import numpy as np

class Linear:
    def __init__(self, input_shape, learning_rate=0.015):
        self.learning_rate = learning_rate
        self.weights = np.random.random(input_shape)
        print(f'weight shape : {self.weights.shape}')
        self.last_output = 0.0

    def run(self, X):
        if X.shape == self.weights.shape:
            self.last_output = sum(X * self.weights)
        else:
            print(f'X is not same shape {X.shape} as weights {self.weights.shape}!')

        return self.last_output

    def learn(self, true_y, x):
        """
        Adjust weights based on last output, the true value of y, and our input
        """
        self.weights += self.learning_rate * (true_y - self.last_output) * x

    def fit(self, X, Y, epochs=1):
        for epoch in range(epochs):
            print(f'Epoch {epoch} / {epochs}', end='\r')
            for t, x in enumerate(X):
                #print(f'\tt = {t} / {len(X)}', end='\r')
                self.run(x)
                self.learn(Y[t], x)
            #print(self.weights)
        print(f'Epoch {epochs} / {epochs}')
        
    def predict(self, X):
        output = []
        for x in X:
            output.append(self.run(x))

        return output

def test():
    X = np.array([[1,3,2],
         [1,5,4],
         [2,6,1],
         [8,3,0],
         [5,5,5]])
        
    y = np.array([3,5,6,3,5])

    model = Linear(len(X[0]), learning_rate=0.02)
    model.fit(X, y, epochs=50)

    print(model.predict(np.array([[4,7,1]])))

if __name__ == '__main__':
    test()