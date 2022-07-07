from tkinter.tix import ACROSSTOP
import scipy
import time
import math
import numpy as np
from scipy import special
from scipy.io import savemat
import tensorflow as tf
reuse=tf.compat.v1.AUTO_REUSE
dtype = np.float32

class model():
    def __init__(self, fd, Ts, M, K, Ns, min_p, max_p, p_n, action_num):
        tf.compat.v1.disable_eager_execution()
        self.fd = fd
        self.Ts = Ts
        self.M = M  #no of antennas
        self.K = K  #no of single antenna users
        self.Ns = Ns  #exploration time
        self.state_num = 1
        self.action_num = action_num
        self.s = tf.compat.v1.placeholder(tf.float32, [None, 1], name ='s')
        self.a = tf.compat.v1.placeholder(tf.float32, [None, ((self.M)*(self.K+1))], name ='a')
        self.q_hat = self.create_dqn(self.s, 'dqn')
        self.min_p = min_p
        self.max_p = max_p
        self.p_n = p_n
        self.INITIAL_EPSILON = 0.2
        self.FINAL_EPSILON = 0.0001
        self.max_reward = 0
        self.batch_size = 500
        self.max_episode = 5000
        self.buffer_size = 50000
        self.power_set = np.hstack([np.zeros((1), dtype=np.float32), 1e-3*pow(10., np.linspace(self.min_p, self.max_p, self.action_num-1)/10.)])


    def state_space(self):
        H_set = np.zeros([self.M,self.K,self.Ns], dtype=dtype)
        pho = np.float32(scipy.special.k0(2*np.pi*self.fd*self.Ts))
        H_set[:,:,0] = np.sqrt(0.5*(np.random.randn(self.M, self.K)**2+np.random.randn(self.M, self.K)**2))
        for i in range(1,self.Ns):
            H_set[:,:,i] = H_set[:,:,i-1]*pho + np.sqrt((1.-pho**2)*0.5*(np.random.randn(self.M, self.K)**2+np.random.randn(self.M, self.K)**2))
        H_tilda = np.zeros([self.M,self.K,self.Ns], dtype=dtype)
        for i in range(self.M):
            for j in range(self.K):
                for k in range(self.Ns):
                    H_tilda[i][j][k] = np.random.normal(0.0, abs(H_set[i][j][k]/10))
        return H_set, H_tilda

    def variable_w(self, shape, name = 'w'):
        w = tf.compat.v1.get_variable(name, shape = shape, initializer = tf.compat.v1.truncated_normal_initializer(stddev=0.1))
        return w

    def variable_b(self, shape, initial = 0.01):
        b = tf.compat.v1.get_variable('b', shape = shape, initializer = tf.constant_initializer(initial))
        return b

    def create_dqn(self, s, name, state_num):
        with tf.compat.v1.variable_scope(name + '.0', reuse = reuse):
            w = self.variable_w([state_num, 128])
            b = self.variable_b([128])
            l = tf.nn.relu(tf.matmul(s, w)+b)
        with tf.compat.v1.variable_scope(name + '.1', reuse = reuse):
            w = self.variable_w([128, 64])
            b = self.variable_b([64])
            l = tf.nn.relu(tf.matmul(l, w) + b)
        with tf.compat.v1.variable_scope(name + '.2', reuse = reuse):
            w = self.variable_w([64, self.action_num])
            b = self.variable_b([self.action_num])
            q_hat = tf.matmul(l, w) + b
        return q_hat

    def predict_q(self, s):
        return self.sess.run(self.q_hat, feed_dict={self.s: s})

    def predict_a(self, s):
        q = self.predict_q(s)
        return np.argmax(q, axis = 0)

    def select_action(self, a_hat, episode):
        epsilon = self.INITIAL_EPSILON - episode * (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.max_episode
        random_index = np.array(np.random.uniform(size = (((self.K+1)*(self.M)))) < epsilon, dtype = np.int32)
        random_action = np.random.randint(0, high = self.action_num, size = ((self.K+1)*(self.M)))
        action_set = np.vstack([a_hat, random_action])
        power_index = action_set[random_index, range((self.K+1)*(self.M))] #[M]
        p = self.power_set[power_index] # W
        a = np.zeros((((self.K+1)*(self.M)), self.action_num), dtype = np.float32)
        a[range((self.K+1)*(self.M)), power_index] = 1.
        return p, a 

    def calculate_rate(self, P):
        p_c = np.zeros(self.M)
        p_k = np.zeros((self.M, self.K))
        for i in range(self.M):
            for j in range(self.K):
                p_k[i][j] = P[(i*(self.K))+j]
        for i in range(self.M):
            p_c[i] = P[self.K*self.M+i-1]
        H_set, H_tilda = self.state_space()
        h_k = H_set[:, :, self.count]
        h_tilda = H_tilda[:, :, self.count]
        sigma2 = 1e-3*pow(10., self.p_n/10.)
        min_rc = 100000
        sum_rk=0
        for i in range(self.K):
            num_rc = (h_k[:, i]*p_c).dot((h_k[:, i]*p_c))
            num_rk = (h_k[:, i]*p_k[:, i]).dot((h_k[:, i]*p_k[:, i]))
            denom_1 = (h_k[:, i]*p_k[:, i]).dot((h_k[:, i]*p_k[:, i]))
            denom_2=0
            for j in range(self.K):
                if(i==j):
                    continue
                denom_2 = denom_2 + (h_tilda[:, i]*p_k[:, i]).dot((h_tilda[:, i]*p_k[:, i]))
            denom_rc = denom_1 + denom_2 + sigma2
            denom_rk = denom_2 + sigma2
            min_rc = min(min_rc, np.log2(1 + num_rc/denom_rc))
            sum_rk = sum_rk + np.log2(1 + num_rk/denom_rk)
        reward = min_rc + sum_rk
        return reward

    def step(self, P):
        reward = self.calculate_rate(P)
        self.count = self.count + 1
        H_set_next = self.H_set[:,:,self.count]
        H_tilda_next = self.H_tilda[:, :, self.count]
        s_actor_next = np.zeros((2*(self.M)*(self.K),1))
        for i in range(self.M):
            for j in range(self.K):
                s_actor_next[i*K+j] = H_set_next[i][j]
        for i in range(self.M):
            for j in range(self.K):
                s_actor_next[M*K+i*K+j-1] = H_tilda_next[i][j]
        s_actor_next = np.float32(s_actor_next)
        return s_actor_next, reward

    def reset(self):
        self.count = 0
        self.H_set, self.H_tilda = self.state_space()
        H_set = self.H_set[:, :, self.count]
        H_tilda = self.H_tilda[:, :, self.count]
        P = np.zeros([self.M*(self.K+1)], dtype=dtype)
        s_actor = np.zeros((2*(self.M)*(self.K),1))
        for i in range(self.M):
            for j in range(self.K):
                s_actor[i*K+j] = H_set[i][j]
        for i in range(self.M):
            for j in range(self.K):
                s_actor[M*K+i*K+j-1] = H_tilda[i][j]
        s_actor = np.float32(s_actor)
        return s_actor, P

    def save_params(self):
        dict_name={}
        for var in tf.compat.v1.trainable_variables(): 
            dict_name[var.name]=var.eval()
        savemat(self.weight_file, dict_name)
    
    def train(self, sess):
        self.sess = sess
        tf.compat.v1.global_variables_initializer().run()
        interval = 100
        st = time.time()
        reward_hist = list()
        for k in range(1, self.max_episode+1):  
            reward_dqn_list = list()
            s_actor, _ = self.reset()
            for i in range(int(Ns)-1):
                a = self.predict_a(s_actor)
                p, a = self.select_action(a, k)
                s_actor_next, r = self.step(p)
                s_actor = s_actor_next
                reward_dqn_list.append(r)

            reward_hist.append(np.mean(reward_dqn_list))   # bps/Hz per link
            if k % interval == 0: 
                reward = np.mean(reward_hist[-interval:])
                if reward > max_reward:
                    model.save_params()
                    max_reward = reward
                print("Episode(train):%d  DQN: %.3f  Time cost: %.2fs" 
                    %(k, reward, time.time()-st))
                st = time.time()
        return reward_hist


if __name__ == "__main__":
    fd = 10
    Ts = 20e-3
    M = 2   
    K = 2
    Ns = 11
    max_p = 38. #dBm
    min_p = 1
    p_n = -114. #dBm
    state_num = (2*(M)*(K))
    action_num = 6  #action_num

    weight_file = 'dqn_6.mat'
    mod = model(fd, Ts, M, K, Ns, min_p, max_p, p_n, action_num)
    global graph
    graph = tf.compat.v1.get_default_graph()
    with graph.as_default():
        with tf.compat.v1.Session() as sess:
            train_hist = mod.train(sess)



