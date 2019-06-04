import numpy as np;
from dataset.mnist import load_mnist;
from TwoLayer import TwoLayerNet;

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True);

train_loss_list = [];

#인간이 정해주는 하이퍼 파라미터
iters_num = 10; #반복횟수
train_size = x_train.shape[0]; #트레이닝 테이더 크기
batch_size = 100; #미니배치 크기
learning_rate = 0.1; #학습율 (이만큼씩 기울기에 곱해진행)

net = TwoLayerNet(input_size=784, hidden_size=50, output_size=10);

for i in range(iters_num):
    #전체 데이터에서 배치 크기만큼 샘플링
    batch_mask = np.random.choice(train_size, batch_size); 
    x_batch = x_train[batch_mask];
    t_batch = t_train[batch_mask]; 

    #샘플링된 데이터를 기반으로 기울기 계산
    grad = net.numerical_gradient(x_batch, t_batch)

    #매개변수를 갱신
    for key in ('W1', 'b1', 'W2', 'b2') :
        #기울기가 minus면 더해질것이고 플러스면 빼진다 (경사하강법)
        net.params[key] -= learning_rate * grad[key]; 

        #학습 경과 기록
        loss = net.loss(x_batch, t_batch);
        train_loss_list.append(loss);
    
print (train_loss_list);