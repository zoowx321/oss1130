import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Dataset loading
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)

# Set up model
x = tf.placeholder(tf.float32, [None, 784])#float32형으로 784차원으로 tensorflow에 input(x)을 주겠다.
#varialbes는 Session에 사용되기전에 반드시 초기화되어야함.
W = tf.Variable(tf.zeros([784, 10]))#W는 784입력과 10개의 출력이 있으므로 784*10행렬을 0으로 초기화
b = tf.Variable(tf.zeros([10]))#b에 0으로 초기화된 10차원 벡터
y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])#y_는 2d 텐서로 float32형으로 10차원 벡터

cross_entropy = -tf.reduce_sum(y_*tf.log(y))#cross_entropy에 y_d의 각 원소들에 각각에 해당하는 y의 원소값에 로그를 취하여 곱하고 tf.reuce_sum을 통해 모든 원소를 더한다.
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)#합습도 0.01로 gradient descent 알고리즘을 이용하여 교차 엔트로피를 최소화

# Session
init = tf.initialize_all_variables()#만든 모든 변수 초기화

sess = tf.Session()#모델을 시작
sess.run(init)#변수 초기화 실행

# Learning
#각 반복단계마다 train set으로 부터 100개의 무작위 데이터들의 일괄 처리들을 저장. 처리한 데이터에 train_step 피딩을 실
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Validation
#모델이 정확한지 평가
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))#y축을 따라 가장 큰 원소의 색인을 구해 y_추에 대한것과 일치하는지 확인
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))#부동소숫점으로 캐스팅 한후 평균값을 구함

# Result should be approximately 91%.
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))#테스트 데이터ㅓ 대상으로 정확도를 확인
