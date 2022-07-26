from tkinter.tix import ACROSSTOP
import scipy
import time
import numpy as np
from scipy.io import savemat
from scipy.io import loadmat
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
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
        self.state_num = 2
        self.action_num = action_num
        self.s = tf.compat.v1.placeholder(tf.float32, [None, 2], name ='s')
        self.a = tf.compat.v1.placeholder(tf.float32, [None, self.action_num], name ='a')
        self.s_c = tf.compat.v1.placeholder(tf.float32, [None, 2*self.K], name ='s_c')
        self.a_c = tf.compat.v1.placeholder(tf.float32, [None, self.action_num], name ='a_c')
        self.q_hat = self.create_dqn(self.s, 'dqn', self.state_num)
        self.qc_hat = self.create_dqn(self.s_c, 'dqn_c', 2*self.K)
        self.y = tf.compat.v1.reduce_sum(tf.multiply(self.q_hat,self. a))
        self.y_c = tf.compat.v1.reduce_sum(tf.multiply(self.qc_hat, self.a_c))
        self.r = tf.compat.v1.placeholder(tf.float32, [None])
        self.loss = tf.nn.l2_loss(self.y - self.r)
        self.loss_c = tf.nn.l2_loss(self.y_c - self.r)
        self.min_p = min_p
        self.max_p = max_p
        self.p_n = p_n
        self.weight_file = 'dqn_6.mat'
        self.INITIAL_EPSILON = 0.2
        self.FINAL_EPSILON = 0.0001
        self.max_reward = 0
        self.max_episode = 5000
        self.buffer_size = 50000
        self.num = 0
        self.batch_size = 32
        self.params, self.params_c = self.get_params('dqn')
        self.learning_rate = 1e-3
        self.power_set = np.hstack([np.zeros((1), dtype=np.float32), 1e-3*pow(10., np.linspace(np.sqrt(self.min_p), np.sqrt(self.max_p), self.action_num-1)/10.)])
        with tf.compat.v1.variable_scope('opt_dqn', reuse = reuse):
            self.optimize = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list = self.params)
            self.optimize_c = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss_c, var_list = self.params_c)

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
    
    def predict_qc(self, s_c):
        return self.sess.run(self.qc_hat, feed_dict={self.s_c: s_c})

    def predict_a(self, s, s_c):
        q = self.predict_q(s)
        q_c = self.predict_qc(s_c)
        q = np.vstack((q, q_c))
        q = np.argmax(q, axis=1)
        return np.float32(q)

    def predict_p(self, s, s_c):
        return self.power_set[(self.predict_a(s, s_c)).astype(int)] 

    def get_params(self, para_name):
        sets=[]
        sets_c = []
        i=0
        for var in tf.compat.v1.trainable_variables():
            if not var.name.find(para_name):
                if(i>5):
                    sets_c.append(var)
                else:
                    sets.append(var)
                i=i+1
        return sets, sets_c

    def select_action(self, a_hat, episode):
        epsilon = self.INITIAL_EPSILON - episode * (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.max_episode
        random_index = np.array(np.random.uniform(size = (((self.K+1)*(self.M)))) < epsilon, dtype = np.int32)
        random_action = np.random.randint(0, high = self.action_num, size = ((self.K+1)*(self.M)))
        action_set = np.vstack([a_hat, random_action])
        power_index = action_set[random_index, range((self.K+1)*(self.M))] #[M]
        power_index = power_index.astype(int)
        p = self.power_set[power_index] # W
        if(np.dot(p.T, p) > self.max_p):
            p = np.float32(p*np.sqrt(self.max_p))/np.float32(np.sqrt(np.dot(p.T, p)))
        a = np.zeros((((self.K+1)*(self.M)), self.action_num), dtype = np.float32)
        a[range((self.K+1)*(self.M)), power_index] = 1
        a_c = a[self.M*self.K : self.M*(self.K+1)]
        a = a[0 : self.M*self.K]
        a = np.float32(a)
        a_c = np.float32(a_c)
        return p, a, a_c 

    def calculate_rate(self, P):
        h_k = self.H_set[:, :, self.count]
        h_tilda = self.H_tilda[:, :, self.count]
        sigma2 = 1e-3*pow(10., self.p_n/10.)
        min_rc = 100000
        sum_rk=0
        for i in range(self.K):
            num_rc = np.dot((h_k[:, i]*P[self.M*self.K : self.M*(self.K+1)]).T, h_k[:, i]*P[self.M*self.K : self.M*(self.K+1)])
            num_rk = np.dot((h_k[:, i]*P[self.M*i : self.M*(i+1)]).T, h_k[:, i]*P[self.M*i : self.M*(i+1)])
            denom_1 = num_rk
            denom_2=0
            for j in range(self.K):
                if(i==j):
                    continue
                denom_2 = denom_2 + np.dot((h_tilda[:, i]*P[self.M*j : self.M*(j+1)]).T, h_tilda[:, i]*P[self.M*j : self.M*(j+1)])
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
        s_actor = np.zeros((self.K*self.M, 2))
        for i in range(self.M):
            for j in range(self.K):
                s_actor[i*self.K+j,0] = H_set_next[i, j]
                s_actor[i*self.K+j,1] = H_tilda_next[i, j]
        s_actor_next = np.float32(s_actor)    
        s_c_actor = np.hstack((H_set_next, H_tilda_next))
        return s_actor_next, s_c_actor, reward

    def reset(self):
        self.count = 0
        self.H_set, self.H_tilda = self.state_space()
        H_set = self.H_set[:, :, self.count]
        H_tilda = self.H_tilda[:, :, self.count]
        s_actor = np.zeros((self.K*self.M, 2))
        for i in range(self.M):
            for j in range(self.K):
                s_actor[i*self.K+j,0] = H_set[i, j]
                s_actor[i*self.K+j,1] = H_tilda[i, j]
        s_actor = np.float32(s_actor)
        s_c_actor = np.hstack((H_set, H_tilda))
        return s_actor, s_c_actor

    def save_params(self):
        dict_name={}
        for var in tf.compat.v1.trainable_variables(): 
            dict_name[var.name]=var.eval()
        savemat(self.weight_file, dict_name)

    def dqn(self, s, a, r):
        return self.sess.run(self.optimize, feed_dict={
            self.s: s, self.a: a, self.r: r})
            
    def dqn_c(self, s_c, a_c, r):
        return self.sess.run(self.optimize_c, feed_dict={
            self.s_c: s_c, self.a_c: a_c, self.r: r})     
    
    def train(self, sess):
        self.sess = sess
        tf.compat.v1.global_variables_initializer().run()
        interval = 100
        st = time.time()
        reward_hist = list()
        rewards = list()
        for k in range(1, self.max_episode+1):
            reward_dqn_list = list()
            s_actor, s_c_actor = self.reset()
            for i in range(int(Ns)-1):
                self.num = self.num + 1
                a = self.predict_a(s_actor, s_c_actor)
                p, a, a_c = self.select_action(a, k)
                s_actor_next, s_c_actor_next, r = self.step(p)
                if((k==1) and (i==0)):
                    s = s_actor
                    s_c = s_c_actor
                    action = a
                    action_c = a_c
                s = np.vstack((s, s_actor))
                action = np.vstack((action, a))
                s_c = np.vstack((s_c, s_c_actor))
                action_c = np.vstack((action_c, a_c))
                rewards.append(r)
                s_actor = s_actor_next
                s_c_actor = s_c_actor_next
                reward_dqn_list.append(r)
                if(self.num >= self.batch_size): 
                    self.num = 0 
                    self.dqn(s, action, rewards)
                    self.dqn_c(s_c, action_c, rewards) 
                    s = s_actor
                    s_c = s_c_actor
                    action = a
                    action_c = a_c
                    rewards = list()       
            reward_hist.append(np.mean(reward_dqn_list))   # bps/Hz per link
            if k % interval == 0: 
                reward = np.mean(reward_hist[-interval:])
                if reward > self.max_reward:
                    self.save_params()
                    self.max_reward = reward
                print("Episode(train):%d  DQN: %.3f  Time cost: %.2fs" 
                    %(k, reward, time.time()-st))
            st = time.time()
        return reward_hist

    def test(self, sess):
        max_episode = 100
        self.Ns = int(5e2+1)
        self.sess = sess
        tf.compat.v1.global_variables_initializer().run()
        st = time.time()
        reward_hist = list()
        for k in range(1, max_episode+1):  
            reward_dqn_list = list()
            s_actor, s_c_actor = self.reset()
            for i in range(int(Ns)-1):
                p = self.predict_p(s_actor, s_c_actor)
                s_actor_next, s_c_actor_next, r = self.step(p)
                s_actor = s_actor_next
                s_c_actor = s_c_actor_next
                reward_dqn_list.append(r)
            reward_hist.append(np.mean(reward_dqn_list))   # bps/Hz per link
            print("Episode(test):%d  DQN: %.3f  Time cost: %.2fs" 
                %(k, reward_hist[-1], time.time()-st))
            st = time.time()
        print("Test average rate: %.3f" %(np.mean(reward_hist)))
        return reward_hist


if __name__ == "__main__":
    fd = 10
    Ts = 20e-3
    M = 4
    K = 6
    Ns = 11
    max_p = 38. #dBm
    min_p = 5
    p_n = -114. #dBm
    state_num = 2
    action_num = 10  #action_num

    weight_file = 'dqn_6.mat'
    mod = model(fd, Ts, M, K, Ns, min_p, max_p, p_n, action_num)
    global graph
    graph = tf.compat.v1.get_default_graph()
    with graph.as_default():
        with tf.compat.v1.Session() as sess:
            train_hist = mod.train(sess)
            test_hist = mod.test(sess)
