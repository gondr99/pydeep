#2층 신경망을 통한 기계학습 구현
from common.function import *;
from common.gradient import numerical_gradient;

class TwoLayerNet :
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01) :
        self.params = {};
        
        #1층과 히든층간의 신경망의 행렬과 bias
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size);
        self.params['b1'] = np.zeros(hidden_size);

        #2층과 히든층간의 신경망의 행렬과 bias
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size);
        self.params['b2'] = np.zeros(output_size)
    
    def predict(self, x):
        #신경망 예측
        W1, W2 = self.params['W1'], self.params['W2'];
        b1, b2 = self.params['b1'], self.params['b2'];

        a1 = np.dot(x, W1) + b1;
        z1 = sigmoid(a1);
        a2 = np.dot(z1, W2) + b2;
        y = softmax(a2);

        return y;
    
    def loss(self, x, t):
        #손실함수
        y = self.predict(x);

        return cross_entropy_error(y, t);
    
    def accuracy(self, x, t):
        y = self.predict(x)
        #2차원배열에서 axis를 1로하면 가로방향으로 최대값이 나온 행렬이 나온다.
        y = np.argmax(y, axis=1);
        t = np.argmax(t, axis=1);

        #최대값이 동일한 값을 모두 더하고 사례수 (행수) 로 나누어 정확도를 구한다.
        accuracy = np.sum(y == t) / float(x.shape[0]);
        return accuracy;

    def numerical_gradient(self, x, t):
        # loss_W = W => self.loss(x,t) 와 동일
        loss_W = lambda W: self.loss(x, t);

        #손실함수에 W값과 b값을 넣어서 편미분하여 기울기를 구한다.
        grad = {};
        grad['W1'] = numerical_gradient(loss_W, self.params['W1']);
        grad['b1'] = numerical_gradient(loss_W, self.params['b1']);
        grad['W2'] = numerical_gradient(loss_W, self.params['W2']);
        grad['b2'] = numerical_gradient(loss_W, self.params['b2']);

        return grad;