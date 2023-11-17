#%% Package Loading
import tensorflow as tf
import numpy as np 
import scipy.io as scio

#%% Parameter and coefficient setup
# The code is built based on case9
# Load the optimization model coefficients
bus = scio.loadmat("bus.mat")
branch = scio.loadmat("branch.mat")
gen = scio.loadmat("gen.mat")
gencost = scio.loadmat("gencost.mat")

# Setup the parameters
num_v = 9 # Number of buses/theta
num_c = 9 # Number of branches
num_p = 3 # Number of generators
num_a = num_v+num_p # Number of variables
num_ac = 2*num_c+2*num_p # Number of constraints
num_d = 1 # Number of loosed constraints + 1
num_l = 0.1 # Check the accuracy of matrix calculation
num_g = 0.01 # Gradient for calculation
num_s = 0 # The index of "type 3" bus

# Setup the matrices in the optimization model
br_x = branch['branch'][:,3]
br_f = branch['branch'][:,0]
br_t = branch['branch'][:,1]
cft = np.zeros([num_c, num_v])
bf = np.zeros([num_c, num_v])
for j in range(num_c):
    cft[j,int(br_f[j]-1)] = 1
    cft[j,int(br_t[j]-1)] = -1
    bf[j,int(br_f[j]-1)] = 1/br_x[j]
    bf[j,int(br_t[j]-1)] = -1/br_x[j]
bbus = np.dot(np.transpose(cft), bf) 
mva = 100
Pmax = gen['gen'][:,8]/mva
Pmin = gen['gen'][:,9]/mva
neg_cg = np.zeros([num_v, num_p])
PD = bus['bus'][:,2]   
order = gen['gen'][:,0]
for i in range(num_p):
    neg_cg[int(order[i])-1,i] = -1
Amis = np.hstack((bbus, neg_cg))
bmis = -PD/mva
upt = branch['branch'][:,5]/mva
upf = branch['branch'][:,5]/mva
Vg_new = np.zeros([num_c,num_p])
bf_new = np.hstack((bf, Vg_new))
cg_new = np.eye(num_p)
Pmin_new = np.hstack((np.zeros([num_p,num_v]), -cg_new))
Pmax_new = np.hstack((np.zeros([num_p,num_v]), cg_new))

# Get all the constraints
cont_new = np.hstack((np.transpose(bf_new), np.transpose(-bf_new), np.transpose(Pmin_new), np.transpose(Pmax_new)))
# Get the range values of constraints
val_new = np.hstack((upf, upt, Pmin, Pmax))
# Get the coefficients of objective function
obj = np.hstack((np.zeros([num_v]), -gencost['gencost'][:,5]))

# Get the coefficient matrices for calculating theta
Amis_new = np.delete(Amis,slice(num_v,num_a),axis = 1)
Amis_new = np.delete(Amis_new,num_s,axis = 1)
Amis_new = np.delete(Amis_new,num_s,axis = 0)

#%% Load the sample solution data for training the model
data = np.load("data_case9.npy")

#%% GAN setup
# define the function to initialize parameters
def xavier_init(size): 
    in_dim = size[0] 
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.) 
    return tf.random.normal(shape=size, stddev=xavier_stddev) 

# generate the matrixs of parameters of discriminator
D_W1 = tf.Variable(xavier_init([num_a, 10])) 
D_b1 = tf.Variable(tf.zeros(shape=[10]))
D_W2 = tf.Variable(xavier_init([10, 10])) 
D_b2 = tf.Variable(tf.zeros(shape=[10]))
D_W3 = tf.Variable(xavier_init([10, 1])) 
D_b3 = tf.Variable(tf.zeros(shape=[1]))

# generate the set of all parameters of discriminator
theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

# generate the matrixs of parameters of generator
G_W1 = tf.Variable(xavier_init([num_p, 10])) 
G_b1 = tf.Variable(tf.zeros(shape=[10])) 
G_W2 = tf.Variable(xavier_init([10, 10])) 
G_b2 = tf.Variable(tf.zeros(shape=[10]))
G_W3 = tf.Variable(xavier_init([10, 10])) 
G_b3 = tf.Variable(tf.zeros(shape=[10]))
G_W4 = tf.Variable(xavier_init([10, 10])) 
G_b4 = tf.Variable(tf.zeros(shape=[10]))
G_W5 = tf.Variable(xavier_init([10, num_p])) 
G_b5 = tf.Variable(tf.zeros(shape=[num_p]))

# generate the set of all parameters of generator
theta_G = [G_W1, G_W2, G_W3, G_W4, G_W5, G_b1, G_b2, G_b3, G_b4, G_b5]

# define random noise of matrix[m,n] as the input of G
def sample_Z(o, m): 
    return np.random.uniform(-1., 1., size=[o, m])

# define the generator
def generator(z, G_f, G_a, set_s):
    G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1) 
    G_log_prob = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_pro = tf.nn.relu(tf.matmul(G_log_prob, G_W3) + G_b3)
    G_pro2 = tf.nn.relu(tf.matmul(G_pro, G_W4) + G_b4)
    G_prob = tf.matmul(G_pro2, G_W5) + G_b5
    G_output, G_set = condfunc(G_prob,G_f, G_a, set_s)
    return G_output, G_set

# define the discriminator
def discriminator(x): 
    inputs = tf.cast(x, dtype = tf.float32)
    D_h1 = tf.nn.leaky_relu(tf.matmul(inputs, D_W1) + D_b1)
    D_h2 = tf.nn.leaky_relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_h3 = tf.nn.leaky_relu(tf.matmul(D_h2, D_W3) + D_b3)
    return D_h3

# define the next batch of each iteration
def next_batch(train_data, batch_size, num):
    index = np.array(range(0,num))
    np.random.shuffle(index)
    batch_data = train_data[index[0:batch_size],:]
    return batch_data

#%% Model-Informed selector setup (Set batch size as 50)
# Get required coefficient matrices for calculating constraints
tensor_Amis1 = tf.tile(tf.expand_dims(tf.cast(tf.constant(Amis_new), tf.float32), axis = 0), [50,1,1])
tensor_Amis2 = tf.tile(tf.expand_dims(tf.cast(tf.constant(np.hstack((np.expand_dims(Amis[:,num_s], axis = 1),Amis[:,num_v:num_a]))), tf.float32), axis = 0), [50,1,1])
tensor_Amis = tf.tile(tf.expand_dims(tf.cast(tf.constant(Amis), tf.float32), axis = 0), [50,1,1])
tensor_bmis = tf.expand_dims(tf.tile(tf.expand_dims(tf.cast(tf.constant(bmis), tf.float32), axis = 0),[50,1]), axis = 2)
tensor_Amis1_inverse = tf.tile(tf.expand_dims(tf.cast(tf.constant(np.linalg.pinv(Amis_new)), tf.float32), axis = 0), [50,1,1])

# Define the function
def condfunc(input_f, input_a, input_n, set_s):  

    # Load and concat P_g and theta
    ten_res = tf.expand_dims(tf.concat([tf.zeros([50,1], tf.float32), input_f], axis = 1), axis = 2)
    ten_remain = tf.subtract(tensor_bmis, tf.matmul(tensor_Amis2, ten_res))
    ten_remain_new = tf.slice(ten_remain,[0,num_s+1,0],[50,num_v-num_s-1,1])
    ten_angle = tf.matmul(tensor_Amis1_inverse, ten_remain_new)

    # Define the input data, constraints and values
    ten_cont = tf.cast(tf.constant(cont_new), tf.float32)
    ten_value = tf.cast(tf.constant(val_new), tf.float32)
    input_new = tf.concat([tf.zeros([50,1], tf.float32), tf.squeeze(ten_angle), input_f], axis = 1)

    # Check whether the solution is accurate or not 
    ten_test = tf.abs(tf.subtract(tensor_bmis, tf.matmul(tensor_Amis,tf.expand_dims(input_new, axis = 2))))
    ten_sum = tf.reduce_sum(tf.cast(tf.cast(tf.less(tf.squeeze(ten_test),num_l),tf.int32),tf.float32),1)
    
    # Check whether the samples satisfy the constraints
    cond_results = tf.subtract(ten_value,tf.matmul(input_new, ten_cont)) 
    cond_sum = tf.reduce_sum(tf.cast(tf.cast(tf.greater(cond_results,0.0),tf.int32),tf.float32),1)
    
    # Replace the samples in the saving matrix by generated samples
    save_results = tf.subtract(ten_value,tf.matmul(input_a, ten_cont)) 
    set_save = tf.reduce_sum(tf.cast(tf.cast(tf.greater(save_results,0.0),tf.int32),tf.float32),1)
    ten_test_a = tf.abs(tf.subtract(tensor_bmis, tf.matmul(tensor_Amis,tf.expand_dims(input_a, axis = 2))))
    ten_sum_a = tf.reduce_sum(tf.cast(tf.cast(tf.less(tf.squeeze(ten_test_a),num_l),tf.int32),tf.float32),1) 
    set_save_n = tf.add(tf.cast(tf.cast(tf.greater(ten_sum_a,num_v-1),tf.int32),tf.float32),tf.cast(tf.cast(tf.greater(set_save,num_ac-num_d),tf.int32),tf.float32))
    set_save_sample = tf.tile(tf.expand_dims(tf.cast(tf.cast(tf.greater(set_save_n,num_ac-num_d),tf.int32),tf.float32), axis = 1),[1,num_a])
    set_save_fake = tf.tile(tf.expand_dims(tf.cast(tf.cast(tf.less_equal(set_save_n,num_ac-num_d),tf.int32),tf.float32), axis = 1),[1,num_a])
    input_a_new = tf.add(tf.multiply(set_save_sample,input_a),tf.multiply(set_save_fake,input_new))
    
    # Check whether the objective function value of input_f is higher than input_a
    obj_param = tf.expand_dims(tf.cast(tf.constant(obj), tf.float32), axis = 1)
    obj_diff = tf.subtract(tf.matmul(input_new, obj_param),tf.matmul(input_a_new, obj_param)) 
    obj_sum = tf.reduce_sum(tf.cast(tf.cast(tf.greater(obj_diff,0.0),tf.int32),tf.float32),1)

    # Update the input data by the objective function value
    set_sample_i = tf.tile(tf.expand_dims(tf.cast(tf.cast(tf.greater(obj_sum,0.0),tf.int32),tf.float32), axis = 1),[1,num_a])
    set_fake_i = tf.tile(tf.expand_dims(tf.cast(tf.cast(tf.less_equal(obj_sum,0.0),tf.int32),tf.float32), axis = 1),[1,num_a])
    cond_obj_i = tf.add(tf.multiply(set_sample_i,input_new),tf.multiply(set_fake_i,input_a_new))    

    # Find the intersection of three conditions
    set_cont = tf.add(tf.cast(tf.cast(tf.greater(cond_sum,num_ac-num_d),tf.int32),tf.float32),tf.cast(tf.cast(tf.greater(ten_sum,num_v-1),tf.int32),tf.float32))
    set_desire = tf.cast(tf.cast(tf.greater(set_cont,1.0),tf.int32),tf.float32)

    # Update the input data by the constraints    
    set_initial = tf.add(tf.cast(tf.cast(tf.greater(cond_sum,num_ac-num_d),tf.int32),tf.float32),tf.cast(tf.cast(tf.greater(ten_sum,num_v-1),tf.int32),tf.float32))
    set_sample = tf.tile(tf.expand_dims(tf.cast(tf.cast(tf.greater(set_initial,1.0),tf.int32),tf.float32), axis = 1),[1,num_a])
    set_fake = tf.tile(tf.expand_dims(tf.cast(tf.cast(tf.less_equal(set_initial,1.0),tf.int32),tf.float32), axis = 1),[1,num_a])
    cond_obj_n = tf.add(tf.multiply(set_sample,cond_obj_i),tf.multiply(set_fake,input_a_new))    

    cond_special = tf.multiply(tf.multiply(set_sample, set_save_sample),set_fake_i)
    cond_obj_p = tf.add(tf.multiply(cond_special,cond_obj_i),tf.multiply(tf.abs(tf.ones([50,num_a])-cond_special),cond_obj_n)) 

    set_s_update = tf.subtract(set_desire, set_s)
    set_s_sample = tf.tile(tf.expand_dims(tf.cast(tf.cast(tf.greater_equal(set_s_update,0.0),tf.int32),tf.float32), axis = 1), [1, num_a])
    set_s_fake = tf.tile(tf.expand_dims(tf.cast(tf.cast(tf.less(set_s_update,0.0),tf.int32),tf.float32), axis = 1), [1, num_a])
    cond_obj = tf.add(tf.multiply(set_s_sample,cond_obj_p),tf.multiply(set_s_fake,input_a))  
    
    # Compare the objective function value of input_f and input_n
    obj_diff_new = tf.subtract(tf.matmul(cond_obj, obj_param),tf.matmul(input_n, obj_param))
    obj_sum_new = tf.reduce_sum(tf.cast(tf.cast(tf.greater(obj_diff_new,0.0),tf.int32),tf.float32),1)
    obj_m = tf.tile(tf.expand_dims(tf.subtract(tf.add(obj_sum_new,obj_sum_new),tf.ones([50,])), axis = 1), [1,num_a])

    # Calculate the parts that need to add gradient
    cond_m = tf.cast(tf.cast(tf.greater(tf.subtract(cond_obj,input_n),0.0),tf.int32),tf.float32)
    cond_m_new = tf.subtract(tf.add(cond_m,cond_m),tf.ones([50,num_a]))
    cond_gradient = tf.concat([tf.zeros([50,num_v]), tf.multiply(tf.ones([50,num_p]),num_g)], axis = 1)
    cond_all = tf.add(tf.multiply(tf.multiply(obj_m,cond_m_new), cond_gradient),cond_obj)

    # Calculate the theta after adding the gradient
    cond_input = tf.slice(cond_all, [0,num_v],[50,num_p])

    # Combine Pg and theta
    ten_res_new = tf.expand_dims(tf.concat([tf.zeros([50,1], tf.float32), cond_input], axis = 1), axis = 2)
    ten_remain_d = tf.subtract(tensor_bmis, tf.matmul(tensor_Amis2, ten_res_new))
    ten_remain_new_d = tf.slice(ten_remain_d,[0,num_s+1,0],[50,num_v-num_s-1,1])
    ten_angle_new = tf.matmul(tensor_Amis1_inverse, ten_remain_new_d)
    cond_new = tf.concat([tf.zeros([50,1], tf.float32), tf.squeeze(ten_angle_new), cond_input], axis = 1)    
    cond_results_new = tf.subtract(ten_value,tf.matmul(cond_new, ten_cont)) 

    # Update the input data according to feasibility
    cond_sum_new = tf.reduce_sum(tf.cast(tf.cast(tf.greater(cond_results_new,0.0),tf.int32),tf.float32),1)
    ten_test2 = tf.abs(tf.subtract(tensor_bmis, tf.matmul(tensor_Amis,tf.expand_dims(cond_new, axis = 2))))
    ten_sum2 = tf.reduce_sum(tf.cast(tf.cast(tf.less(tf.squeeze(ten_test2),num_l),tf.int32),tf.float32),1)
    set_initial_new = tf.add(tf.cast(tf.cast(tf.greater(cond_sum_new,num_ac-num_d),tf.int32),tf.float32),tf.cast(tf.cast(tf.greater(ten_sum2,num_v-1),tf.int32),tf.float32))
    set_sample_new = tf.tile(tf.expand_dims(tf.cast(tf.cast(tf.greater(set_initial_new,1.0),tf.int32),tf.float32), axis = 1),[1,num_a])
    set_fake_new = tf.tile(tf.expand_dims(tf.cast(tf.cast(tf.less_equal(set_initial_new,1.0),tf.int32),tf.float32), axis = 1),[1,num_a])    
    
    # Get the output
    output = tf.add(tf.multiply(set_fake_new,cond_obj),tf.multiply(set_sample_new,cond_new)) 
        
    return output, set_desire # Return the output and the set showing which solution is feasible
    
#%% Model training
# Define the optimizers
D_optimizer = tf.keras.optimizers.RMSprop(0.0001)
G_optimizer = tf.keras.optimizers.RMSprop(0.0001)

# Define the parameter for model training
it_times = 1000
zdim = num_p # Set the noise dimension
G_fake = np.ones([50, num_a]) * 1000 # Set the initial saved solution set
c_set = np.zeros([50]) # Set the initial set to show the feasiblity of saved solution set

# Train the model and get the solutions
for it in range(it_times):
    X = next_batch(data, 50, data.shape[0])
    Z = sample_Z(50,zdim)
    
    c_set = tf.convert_to_tensor(c_set)
    G_fake = tf.convert_to_tensor(G_fake)
    X = tf.convert_to_tensor(X)
    Z = tf.convert_to_tensor(Z)
    c_set = tf.cast(c_set, tf.float32)
    G_fake = tf.cast(G_fake, tf.float32)
    X = tf.cast(X, tf.float32)
    Z = tf.cast(Z, tf.float32)
    
    with tf.GradientTape() as D_tp, tf.GradientTape() as G_tp:
        # Get the result of generator
        G_fake, set_c = generator(Z, G_fake, X, c_set)
        # Get the probability of real and fake samples separately
        D_real = discriminator(X)
        D_fake = discriminator(G_fake) 

        c_set = 1 - (1 - c_set) * (1 - set_c)
        
        D_loss = tf.reduce_mean(D_fake) -tf.reduce_mean(D_real) 
        G_loss = -tf.reduce_mean(D_fake)
        		
        D_gradients = D_tp.gradient(D_loss, theta_D)
        G_gradients = G_tp.gradient(G_loss, theta_G)
        D_optimizer.apply_gradients(zip(D_gradients, theta_D))
        G_optimizer.apply_gradients(zip(G_gradients, theta_G)) 

#%% Get the desired solution
# If recuriver iteration algorithm is needed, select and save the generated data and original data, then re-run the Model training part
select_sample = np.squeeze(np.where(c_set == 1)) 
select_sample_best = np.where(np.dot(G_fake.numpy(), np.expand_dims(obj,axis=1)) == np.max(np.dot(G_fake.numpy()[np.squeeze(np.where(c_set == 1)),:], np.expand_dims(obj,axis=1))))[0]
print("Desired solution:", G_fake.numpy()[select_sample_best,:])