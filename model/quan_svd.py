## here my own model will be implemented, with tensorflow
## this implementation follow the style of https://github.com/tensorflow/models/blob/master/tutorials/embedding/word2vec.py
import numpy as np
import tensorflow as tf



## which is used to specify some parameters
flags = tf.app.flags


## 671 + 1, 9066 + 1
## 77805 + 1, 185973 + 1
## 383033 + 1, 80008 + 1
## 1.8206


# currently, we only use the one for movielens
flags.DEFINE_integer("num_u", 671+1, "number of operators in the sys")
flags.DEFINE_integer("num_v", 9066+1, "number of pts in the sys")
flags.DEFINE_integer("embed_dim", 5, "number of embedding dimension");
flags.DEFINE_integer("batch_size", 64, "the size of a mini-batch");
flags.DEFINE_float("fuzziness", 1.2, "a value greater than 1 to prevent the model from attaining a trivial solution")
flags.DEFINE_integer("u_num_state", 10, "the eigen-state of each user")
flags.DEFINE_integer("v_num_state", 10, "the eigen-state of each item")
flags.DEFINE_float("mean", 3.0, "a kind of initial guess of the rating")

flags.DEFINE_float("learning_rate", 0.05, "learning rate");
flags.DEFINE_integer("max_epoch", 20, "the maximum iteration over the whole dataset");
flags.DEFINE_integer("test_batch_size", 1000, "the batch size for a distributed calculation of the accuracy")
flags.DEFINE_float("reg_all", 0.05, "regularization coeff");
FLAGS = flags.FLAGS;

class Options(object):
    def __init__(self):
        self.num_u = FLAGS.num_u;
        self.num_v = FLAGS.num_v;
        self.fuzziness = FLAGS.fuzziness;
        self.embed_dim = FLAGS.embed_dim;
        self.batch_size = FLAGS.batch_size;
        self.u_num_state = FLAGS.u_num_state;
        self.v_num_state = FLAGS.v_num_state;
        self.mean = FLAGS.mean;
        self.learning_rate = FLAGS.learning_rate;
        self.max_epoch = FLAGS.max_epoch;
        self.test_batch_size = FLAGS.test_batch_size;
        self.reg_all = FLAGS.reg_all;
        
class QuanSVD(object):
    ## init the model with given parameters
    def __init__(self, options, session, feeder, test_feeder):
        self._options = options;
        self._session = session;
        self.feeder = feeder;
        self.test_feeder = test_feeder;
        self.build_graph();

    ## the ids should be of [batch_size, 2], integer
    ## the rs should be of [batch_size]
    def forward(self, ids, rs):
        opts = self._options;
        
        init_width = 0.5 / opts.embed_dim;

        ## create the embedding table
        u_emb = tf.Variable(
            tf.random_normal([opts.num_u, opts.u_num_state, opts.embed_dim])
        );
        v_emb = tf.Variable(
            tf.random_normal([opts.num_v, opts.v_num_state, opts.embed_dim])
        );

        
        ## create the probability
        if(opts.u_num_state != 1):
            u_prob_free = tf.Variable(
                tf.random_uniform([opts.num_u, opts.u_num_state - 1], -init_width, init_width)
            );
        else:
            u_prob_free = tf.Variable(tf.ones([opts.num_u, opts.u_num_state]));
            
        if(opts.v_num_state != 1):
            v_prob_free = tf.Variable(
                tf.random_uniform([opts.num_v, opts.v_num_state - 1], -init_width, init_width)
            );
        else:
            v_prob_free = tf.Variable(tf.ones([opts.num_v, opts.v_num_state]));

        u_bias = tf.Variable(
            tf.zeros([opts.num_u, 1])
        );

        v_bias = tf.Variable(
            tf.zeros([opts.num_v, 1])
        );


        ## to register the parameters in this class
        self._u_emb = u_emb;
        self._v_emb = v_emb;
        self._u_p_free = u_prob_free;
        self._v_p_free = v_prob_free;
        self._u_bias = u_bias;
        self._v_bias = v_bias;
        
        
        
        ## do the embedding according to the sampled ids in a batch
        sampled_u_emb = tf.nn.embedding_lookup(u_emb, ids[:, 0]); # of size [batch_size, u_num_state, embed_dim]
        sampled_v_emb = tf.nn.embedding_lookup(v_emb, ids[:, 1]); # of size [batch_size, v_num_state, embed_dim]

        ## calculate the probablity of each u and v falling in per eigen-state
        if(opts.u_num_state != 1):
            sampled_u_p = self.calc_prob(tf.nn.embedding_lookup(u_prob_free, ids[:, 0])); # of size [batch_size, u_num_state]
        else:
            sampled_u_p = tf.nn.embedding_lookup(u_prob_free, ids[:, 0]);

        if(opts.v_num_state != 1):
            sampled_v_p = self.calc_prob(tf.nn.embedding_lookup(v_prob_free, ids[:, 1])); # of size [batch_size, v_num_state]
        else:
            sampled_v_p = tf.nn.embedding_lookup(v_prob_free, ids[:, 1]);
        
        sampled_u_bias = tf.expand_dims(tf.nn.embedding_lookup(u_bias, ids[:, 0]), 2); ## [batch_size, 1]
        sampled_v_bias = tf.expand_dims(tf.nn.embedding_lookup(v_bias, ids[:, 1]), 2); ## [batch_size, 1]
        
        ones_dims_A = tf.stack([tf.shape(self._ids)[0], opts.u_num_state, 1]);
        ones_dims_B = tf.stack([tf.shape(self._ids)[0], 1, opts.v_num_state]);
        sampled_u_bias = tf.matmul(tf.fill(ones_dims_A, 1.0), tf.matmul(sampled_u_bias, tf.fill(ones_dims_B, 1.0)));
        sampled_v_bias =  tf.matmul(tf.fill(ones_dims_A, 1.0), tf.matmul(sampled_v_bias, tf.fill(ones_dims_B, 1.0)));
        ## compute the prediction
        preds = opts.mean + sampled_u_bias + sampled_v_bias + tf.matmul(sampled_u_emb, tf.transpose(sampled_v_emb, perm = [0, 2, 1]));
        # print(tf.shape(sampled_v_p));
        return tf.expand_dims(sampled_u_p, 2), tf.expand_dims(sampled_v_p, 2), preds;
        

    """ tool functions """
    def calc_prob(self, p_free):
        opts = self._options;
        p_tan = tf.exp(p_free); # (batch_size, num_state - 1)
        ones_dims = tf.stack([tf.shape(self._ids)[0], 1]);
        p_tan = tf.concat([p_tan, tf.fill(ones_dims, 1.0)], axis = 1); # [batch_size, num_state]
        sums = tf.reduce_sum(p_tan, axis = 1, keepdims = True); # (batch_size, 1)
        return tf.div(p_tan, tf.matmul(sums, tf.ones([1, tf.shape(p_tan)[1]])));

    ## work as a wrapper for creating an optimizer
    def optimize(self, loss):
        opts = self._options;
        lr = opts.learning_rate;
        optimizer = tf.train.MomentumOptimizer(lr, 0.005);
        _var_list = [self._u_emb, self._v_emb, self._u_bias, self._v_bias];
        if(opts.u_num_state != 1):
            _var_list.append(self._u_p_free);
        if(opts.v_num_state != 1):
            _var_list.append(self._v_p_free);
        train = optimizer.minimize(loss, var_list = _var_list);
        self._train = train;

    def reg_term(self):
        opts = self._options;
        return  (tf.norm(self._u_emb) + tf.norm(self._v_emb));

    ## define the loss for the model to optimize
    def quan_l2_loss(self, u_p, v_p, preds, rs):
        """
        u_p [b, u_num_state, 1] v_p [b, v_num_state, 1] pres [b, u_num_state, v_num_state] rs [b]
        """
        opts = self._options;
        prob_mat = tf.matmul(u_p, tf.transpose(v_p, perm = [0, 2, 1])); # [b, u_num_state, v_num_state]
        prob_mat = tf.pow(prob_mat, opts.fuzziness);
        rs = tf.expand_dims(tf.expand_dims(rs, 1), 2);
        ones_dims_A = tf.stack([tf.shape(self._ids)[0], opts.u_num_state, 1]);
        ones_dims_B = tf.stack([tf.shape(self._ids)[0], 1, opts.v_num_state]);
        rs_mat = tf.matmul(tf.fill(ones_dims_A, 1.0), tf.matmul(rs, tf.fill(ones_dims_B, 1.0)));
        
        return tf.reduce_sum(tf.multiply(prob_mat, tf.pow(preds - rs_mat ,2)));

    def predict(self, u_p, v_p, preds):
        opts = self._options;
        prob_mat = tf.matmul(u_p, tf.transpose(v_p, perm = [0, 2, 1])); # [b, u_num_state, v_num_state] 
        return tf.reduce_sum(tf.multiply(prob_mat, preds), axis = [1,2]);
    
        
    def build_graph(self):
        opts = self._options;
        ids = tf.placeholder(tf.int32, shape = (None, 2));
        rs = tf.placeholder(tf.float32, shape = (None));
        self._ids = ids;
        self._rs = rs;
        (u_p, v_p, preds_m) = self.forward(ids, rs);
        loss = self.quan_l2_loss(u_p, v_p, preds_m, rs) + opts.reg_all * self.reg_term();
        preds = self.predict(u_p, v_p, preds_m);

        self.loss = loss;
        self.preds = preds;
        ## append the optimizer to the graph
        self.optimize(loss);
        # init all the variables
        tf.global_variables_initializer().run();

    def test(self):
        ## directly return the RMSE first
        opts = self._options;
        RMSE = 0.0;
        MAE = 0.0;
        self._session.run(self.test_feeder.iterator.initializer);
        while True:
            try:
                batch = self.test_feeder.next_batch();
                _preds = self._session.run([self.preds], {
                    self._ids : batch[:, 0:2].astype(np.int32)
                })
                _preds = np.array(_preds).T
                # print(np.random.choice(_preds[:,0], 4));
                RMSE += np.sum((batch[:, 2] - _preds[:,0]) ** 2) / self.test_feeder.test_size;
                MAE += np.sum(np.abs(batch[:, 2] - _preds[:, 0])) / self.test_feeder.test_size;
            except tf.errors.OutOfRangeError:
                break;
        return np.sqrt(RMSE), MAE;
                
                
        
        


    def train(self):
        opts = self._options;
        print("(U_STATE:{}, V_STATE:{})".format(opts.u_num_state, opts.v_num_state));
        avg_epoch_loss = 0.0;
        min_rmse = 1000;
        min_mae = 1000;
        for epoch_n in range(opts.max_epoch):
            # print("BEGIN EPOCH {}".format(epoch_n + 1));
            self._session.run(self.feeder.iterator.initializer);
            rmse, mae = self.test();
            if(rmse <= min_rmse):
                min_rmse = rmse;
            if(mae <= min_mae):
                min_mae = mae;
            print("EPOCH {} AVG LOSS {} RMSE {} MAE {}".format(epoch_n , avg_epoch_loss, rmse, mae));
            avg_epoch_loss = 0.0;
            j = 0;
            while True:
                j += 1;
                try:
                    batch = self.feeder.next_batch();
                    _loss = self._session.run([self.loss], {
                        self._ids: batch[:,0:2].astype(np.int32),
                        self._rs : batch[:,2].astype(np.float32)
                    })
            
                    self._session.run([self._train],{
                        self._ids: batch[:,0:2].astype(np.int32),
                        self._rs : batch[:,2].astype(np.float32)
                    });
                    avg_epoch_loss += np.mean(_loss);
                    # if(j % 500 == 0):
                        # print("LOG {}".format(j));
                except tf.errors.OutOfRangeError:
                    break
            avg_epoch_loss = avg_epoch_loss / j;
        print("(U {}, V {}): (RMSE {}, MAE {})".format(opts.u_num_state, opts.v_num_state, min_rmse, min_mae));
        

    
## for data feed, based on tf.data.Dataset
class Feeder(object):
    def __init__(self, session, path, batch_size):
        ds = np.load(path);
        self.dataset = tf.data.Dataset.from_tensor_slices(ds).shuffle(1000).batch(batch_size);
        self.batch_size = batch_size;
        self.session = session;
        self.iterator = self.dataset.make_initializable_iterator();
        self.next_element = self.iterator.get_next();
        
    ## next random batch
    def next_batch(self):
        return self.session.run(self.next_element);

class TestFeeder(object):
    def __init__(self, session, path, batch_size):
        ds = np.load(path);
        self.test_size = len(ds);
        self.dataset = tf.data.Dataset.from_tensor_slices(ds).batch(batch_size);
        self.batch_size = batch_size;
        self.session = session;
        self.iterator = self.dataset.make_initializable_iterator();
        self.next_element = self.iterator.get_next();

    ## next ordered batch
    def next_batch(self):
        return self.session.run(self.next_element);
        
        
        
    
    
def main(_):
    opts = Options();
    PREFIX = "/home/mlsnrs/data/pxd/paper4graduation/paper_exp/dataset/";
    names = [ 'ml-latest-small/ml', 'BX-CSV-Dump/bx', 'jester/jester', 'douban/douban'];
    teller = ["MovieLens", "BookCrossing", "Jester", "Douban"];

    # opts.u_num_state  = u_num_state;
    # opts.v_num_state = v_num_state;
 
    with tf.Graph().as_default(), tf.Session() as session:
        feeder = Feeder(session, PREFIX + names[0] + ".train", opts.batch_size);
        test_feeder = TestFeeder(session, PREFIX + names[0] + ".test", opts.test_batch_size);
        model = QuanSVD(opts, session, feeder, test_feeder);
        model.train();

if __name__ == "__main__":  
    tf.app.run();
        

    


