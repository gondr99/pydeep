import numpy as np

#계층별 뉴런과 bias값 정의
def init_network() :
    network = {};
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]);
    network['b1'] = np.array([0.1, 0.2, 0.3]);
    network['W2'] = np.array([[0.1, 0.4],[0.2, 0.5],[0.3, 0.6]]);
    network['b2'] = np.array([0.1, 0.2]);
    network['W3'] = np.array([[0.1,0.3], [0.2,0.4]]);
    network['b3'] = np.array([0.1, 0.2]);
    return network;

# 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x));

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3'];
    b1, b2, b3 = network['b1'], network['b2'], network['b3'];

    # numpy의 브로드캐스트 기능으로 인해 A1의 각 원소가 하나씩 맵핑되서 함수가 실행된다.    
    a1 = np.dot(x, W1) + b1;
    z1 = sigmoid(a1);
    a2 = np.dot(z1, W2) + b2;
    z2 = sigmoid(a2);
    a3 = np.dot(z2, W3) + b3;
    #항등함수는 생략

    return a3

# 입력값
x = np.array([1.0, 0.5]);
network = init_network();
result = forward(network, x);

print(result);