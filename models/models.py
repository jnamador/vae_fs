"""The purpose of this script is to hold the VAE Model class defintion and 
related functions to keep the notebooks cleaner"""


import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.regularizers import l1_l2

class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    

def Qmake_encoder_set_weights(input_dim,h_dim_1,h_dim_2,latent_dim):
    """
    Makes encoder

    Parameters
    ----------
    input_dim : int
        size of input layer
    h_dim_[X] : int
        size of hidden layer X
    latent_dim : int
        size of latent layer
    """

    # What is this and why? ----------------------------------------------------
    # update: well we don't want to be too different from Kenny's repo afterall. Initialization in layers are kept to stay consistent with Kenny. Batch normalization removed for same reason.
    l2_factor = 1e-3 
    # --    

    # Input layer
    inputs = keras.Input(shape=(input_dim))

    # Hidden layer 1 -----------------------------------------------------------
    x = Dense(h_dim_1,
             kernel_initializer=keras.initializers.HeNormal(seed=None), 
             bias_initializer=keras.initializers.Zeros(),
             kernel_regularizer=l1_l2(l1=0, l2=l2_factor), # This is where the l2_factor is used.
             name = "enc_dense1")(inputs)
    x = LeakyReLU(name="enc_Lrelu1")(x)
    # ---

    # Hidden Layer 1 -----------------------------------------------------------
    x = Dense(h_dim_2,
             kernel_initializer=keras.initializers.HeNormal(seed=None),
             bias_initializer=keras.initializers.Zeros(),
             kernel_regularizer=l1_l2(l1=0, l2=l2_factor),
             name = "enc_dense2")(x)
    x = LeakyReLU(name="enc_Lrelu2")(x)
    # ---

    # Latent layer -------------------------------------------------------------
    # No activation. 
    z_mean=Dense(latent_dim, name='z_mean',
                  kernel_initializer=keras.initializers.HeNormal(seed=None),
                  bias_initializer=keras.initializers.Zeros(),
                  kernel_regularizer=l1_l2(l1=0, l2=l2_factor)
                )(x)
    z_logvar=Dense(latent_dim, name='z_log_var',
                      kernel_initializer=keras.initializers.Zeros(),
                      bias_initializer=keras.initializers.Zeros(),
                      kernel_regularizer=l1_l2(l1=0, l2=l2_factor)
                    )(x)
     # Comparing this to eq 2 in arXiv: 2108.03986 z_log_var = log(sigma**2)
    z=Sampling()([z_mean,z_logvar])
    # ---


    encoder = keras.Model(inputs,[z_mean,z_logvar,z],name='encoder')
    return encoder


def Qmake_decoder_set_weights(input_dim,h_dim_1,h_dim_2,latent_dim):
    """ 
    Makes decoder

    Parameters
    ----------
    input_dim : int
        size of input layer
    h_dim_[X] : int
        size of hidden layer X
    latent_dim : int
        size of latent layer
    """
    l2_factor = 1e-3
    # Input layer -------
    inputs=keras.Input(shape=(latent_dim)) 

    # Hiden layer 1 (3 total, not counting latent) -------
    x = Dense(h_dim_2,
                   kernel_initializer=keras.initializers.HeNormal(seed=None),
                   bias_initializer=keras.initializers.Zeros(),
                   kernel_regularizer=l1_l2(l1=0, l2=l2_factor)
                   )(inputs)
    x = LeakyReLU(name="dec_Lrelu3")(x)
    # --


    # Hidden layer 2( 4 total, not counting laten) -----
    x = Dense(h_dim_1,
    # ? ----  #    activation='relu', # Why ReLU over papers leaky ReLU?
                   kernel_initializer=keras.initializers.HeNormal(seed=None),
                   bias_initializer=keras.initializers.Zeros(),
                   kernel_regularizer=l1_l2(l1=0, l2=l2_factor)
                   )(x)
    x = LeakyReLU(name="dec_Lrelu4")(x)
    # --

    x = Dense(input_dim,
                   kernel_initializer=keras.initializers.HeNormal(seed=None),
                   bias_initializer=keras.initializers.Zeros(),
                   kernel_regularizer=l1_l2(l1=0, l2=l2_factor)
                   )(x)
    y = LeakyReLU(name="dec_Lrelu5")(x)
    decoder=keras.Model(inputs, y,name='decoder')
    return decoder

def _custom_MSE(masked_data, masked_reconstruction):
#     # "We use a dataset with standardized p_T as a target so that all quantities are O(1)" arXiv: 2108.03986 

#     # Q: is the input also standardized?
    

#     jet_scale = 256/64
#     tau_scale = 128/64
#     muon_scale = 32/64
#     met_scale = 512/64
#     em_scale = 128/64
    jet_scale = 1
    tau_scale = 1
    muon_scale = 1
    met_scale = 1
    em_scale = 1
    # Define the indices and their corresponding scale factors
    scale_dict = {
        0: met_scale,
        3: em_scale, 6: em_scale, 9: em_scale, 12: em_scale,
        15: tau_scale, 18: tau_scale, 21: tau_scale, 24: tau_scale,
        27: jet_scale, 30: jet_scale, 33: jet_scale, 36: jet_scale, 39: jet_scale, 42: jet_scale,
        45: muon_scale, 48: muon_scale, 51: muon_scale, 54: muon_scale
    }

    # Create the scaling tensor
    scale_tensor = tf.ones_like(masked_data)
    for index, factor in scale_dict.items():
        index_mask = tf.one_hot(index, depth=tf.shape(masked_data)[-1])
        scale_tensor += index_mask * (factor - 1)

    # Apply scaling
    scaled_data = masked_data * scale_tensor
    scaled_reconstruction = masked_reconstruction * scale_tensor

    # Hardcoded lists for eta and phi indices
    eta_indices = [4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55]
    phi_indices = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47, 50, 53, 56]

    batch_size = tf.shape(scaled_reconstruction)[0]
    
    # Set only the first eta (index 1) to zero
    indices = tf.stack([tf.range(batch_size), tf.ones(batch_size, dtype=tf.int32)], axis=1)
    updates = tf.zeros(batch_size)
    scaled_reconstruction = tf.tensor_scatter_nd_update(scaled_reconstruction, indices, updates)
    
    # Apply constraints to eta
    for i in eta_indices:
        indices = tf.stack([tf.range(batch_size), tf.fill([batch_size], i)], axis=1)
        updates = 3 * tf.tanh(scaled_reconstruction[:, i] / 3)
        scaled_reconstruction = tf.tensor_scatter_nd_update(scaled_reconstruction, indices, updates)
    
    # Apply constraints to phi
    for i in phi_indices:
        indices = tf.stack([tf.range(batch_size), tf.fill([batch_size], i)], axis=1)
        updates = 3.14159265258979 * tf.tanh(scaled_reconstruction[:, i] / 3.14159265258979)
        scaled_reconstruction = tf.tensor_scatter_nd_update(scaled_reconstruction, indices, updates)
        
    # Calculate MSE using keras.losses.mse
    mse = keras.losses.mse(scaled_data, scaled_reconstruction)

    # Take the sum across all dimensions
    return tf.reduce_mean(mse)

class VAE_Model(keras.Model):
    def __init__(self, encoder, decoder, steps_per_epoch=3125,cycle_length=10, min_beta=0.1, max_beta=0.85, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        # beta tuning part?
        self.cycle_length = tf.cast(cycle_length, tf.float32)
        self.steps_per_epoch = steps_per_epoch
        self.min_beta = tf.cast(min_beta, tf.float32)
        self.max_beta = tf.cast(max_beta, tf.float32)
        self.beta = tf.Variable(min_beta, dtype=tf.float32)
        self.beta_tracker = keras.metrics.Mean(name="beta")

        # per keras VAE example https://keras.io/examples/generative/vae/
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.beta_tracker,
        ]

    def cyclical_annealing_beta(self, epoch):
        # is this the beta tuning?  
        cycle = tf.floor(1.0 + epoch / self.cycle_length)
        x = tf.abs(epoch / self.cycle_length - cycle + 1)
        return self.min_beta + (self.max_beta - self.min_beta) * tf.minimum(x, 1.0)
    

    def train_step(self, data):
        # Is this the beta tuning?
        epoch = tf.cast(self.optimizer.iterations / self.steps_per_epoch, tf.float32)
        
        # Update beta
        self.beta.assign(self.cyclical_annealing_beta(epoch))

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # here we shove in our custom reconstructionn loss function
            # Ignore zero-padded entries. 
            mask = K.cast(K.not_equal(data, 0), K.floatx()) 
            reconstruction_loss = _custom_MSE(mask*reconstruction, mask*data)
            reconstruction_loss *=(1-self.beta)

            # This is just standard Kullback-Leibler diversion loss. I think this can stay.
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *=self.beta
            # Now let solve what beta is
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "beta": self.beta,
        }
    
    # Since we overrode train_step we need test_step
    def test_step(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        
        mask = K.cast(K.not_equal(data, 0), K.floatx())
        reconstruction_loss = _custom_MSE(mask*data, mask*reconstruction)
        reconstruction_loss *= (1 - self.beta)

        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(kl_loss)        
        kl_loss *=self.beta
        
        total_loss = reconstruction_loss + kl_loss
        
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "beta": self.beta,
        }


    def call(self, data):
        z_mean,z_log_var,x = self.encoder(data)
        reconstruction = self.decoder(x)
        return {
            "z_mean": z_mean,
            "z_log_var": z_log_var,
            "reconstruction": reconstruction
        }
    
class VAE_Model_ATLAS_beta(VAE_Model):
    """
    Same as VAE_Model but with the beta cyclical annealing from the 'ATLAS' version.
    """
    def cyclical_annealing_beta(self, epoch):
        cycle = tf.floor(1.0 + epoch / self.cycle_length)
        x = tf.abs(epoch / self.cycle_length - cycle + 1)
        # For first half (x < 0.5), scale 2x from 0 to 1
        # For second half (x >= 0.5), stay at 1
        scaled_x = tf.where(x < 0.5, 2.0 * x, 1.0)
        return self.min_beta + (self.max_beta - self.min_beta) * scaled_x