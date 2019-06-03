import numpy as np;
from PIL import Image;
from dataset.mnist import load_mnist;  #Mnist 데이터를 numpy로 변경해주는 함수
import pickle;

#판정이 아닌 분류를 위한 소프트맥스 함수
def softmax(a):
    c = np.max(a);  #최대값을 빼줘서 지수함수의 오버플로를 막는다.
    exp_a = np.exp(a - c);
    sum_exp_a = np.sum(exp_a);
    return exp_a / sum_exp_a;

# 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x));

#디버깅을 위한 이미지 보여주는 함수
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img));
    pil_img.show();

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True, one_hot_label=False);
    return x_test, t_test;

def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f);
    return network;

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3'];
    b1, b2, b3 = network['b1'], network['b2'], network['b3'];

    a1 = np.dot(x, W1) + b1;
    z1 = sigmoid(a1);
    a2 = np.dot(z1, W2) + b2;
    z2 = sigmoid(a2);
    a3 = np.dot(z2, W3) + b3;

    return softmax(a3);

#데이터와 네트워크 로드 x가 문제 t가 정답
x, t = get_data();
net = init_network();

#배치 처리를 위한 크기
batch_size = 100;
accuracy_cnt = 0;
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size];
    y_batch = predict(net, x_batch);
    
    #가장 높은 점수를 받은 것을 받는다.
    p = np.argmax(y_batch, axis=1); 
    #정답과 일치하면 카운트 증가
    accuracy_cnt += np.sum(p == t[i:i + batch_size]);

print("정확도 : " + str(float(accuracy_cnt) / len(x)));

