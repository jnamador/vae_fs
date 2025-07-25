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
        updates = 3.14159265258979 * tf.tanh(scaled_reconstruction[:, i] / 3.14159265258979) # change to np.pi()
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
            # TODO: add unscaled losses
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
    
# class VAE_Model_ATLAS_beta(VAE_Model):
#     """
#     Same as VAE_Model but with the beta cyclical annealing from the 'ATLAS' version.
#     """
#     def cyclical_annealing_beta(self, epoch):
#         cycle = tf.floor(1.0 + epoch / self.cycle_length)
#         x = tf.abs(epoch / self.cycle_length - cycle + 1)
#         # For first half (x < 0.5), scale 2x from 0 to 1
#         # For second half (x >= 0.5), stay at 1
#         scaled_x = tf.where(x < 0.5, 2.0 * x, 1.0)
#         return self.min_beta + (self.max_beta - self.min_beta) * scaled_x

# VAE-GAN ------------

def Qmake_discriminator(input_dim, h_dim_1, h_dim_2):
    inputs = keras.Input(shape=(input_dim))
    x = Dense(h_dim_1,
              activation='relu',
              kernel_initializer=keras.initializers.HeNormal(seed=None),
              bias_initializer=keras.initializers.Zeros())(inputs)
    x = Dense(h_dim_2,
              activation='relu',
              kernel_initializer=keras.initializers.HeNormal(seed=None),
              bias_initializer=keras.initializers.Zeros())(x)
    x = Dense(1,
              activation='sigmoid',  # Output probability
              kernel_initializer=keras.initializers.HeNormal(seed=None),
              bias_initializer=keras.initializers.Zeros())(x)
    discriminator = keras.Model(inputs, x, name='discriminator')
    return discriminator
def custom_mse_loss_with_multi_index_scaling(masked_data, masked_reconstruction):
    jet_scale = 1
    tau_scale = 1
    muon_sacle = 1
    met_scale = 1

    # Define the indices and their corresponding scale factors
    scale_dict = {
        0: jet_scale,
        3: jet_scale,
        6: jet_scale,
        9: jet_scale,
        12: jet_scale,
        15: jet_scale,
        18: tau_scale,
        21: tau_scale,
        24: tau_scale,
        27: tau_scale,
        30: muon_sacle,
        33: muon_sacle,
        36: muon_sacle,
        39: muon_sacle,
        42: met_scale
    }
    
    # Create the scaling tensor
    scale_tensor = tf.ones_like(masked_data)
    
    for index, factor in scale_dict.items():
        index_mask = tf.one_hot(index, depth=tf.shape(masked_data)[-1])
        scale_tensor += index_mask * (factor - 1)
    
    # Apply scaling
    scaled_data = masked_data * scale_tensor
    scaled_reconstruction = masked_reconstruction * scale_tensor
    
#     # Hardcoded lists for eta and phi indices
#     eta_indices = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40]
#     phi_indices = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 43]

#     batch_size = tf.shape(scaled_reconstruction)[0]
    
#     # Apply constraints to eta
#     for i in eta_indices:
#         indices = tf.stack([tf.range(batch_size), tf.fill([batch_size], i)], axis=1)
#         updates = 3 * tf.tanh(scaled_reconstruction[:, i] / 3)
#         scaled_reconstruction = tf.tensor_scatter_nd_update(scaled_reconstruction, indices, updates)
    
#     # Apply constraints to phi
#     for i in phi_indices:
#         indices = tf.stack([tf.range(batch_size), tf.fill([batch_size], i)], axis=1)
#         updates = 3.14159265258979*(10/8) * tf.tanh(scaled_reconstruction[:, i] / (3.14159265258979*(10/8)))
#         scaled_reconstruction = tf.tensor_scatter_nd_update(scaled_reconstruction, indices, updates)
    # Calculate MSE using keras.losses.mse
    mse = keras.losses.mse(scaled_data, scaled_reconstruction)
    
    # Take the mean across all dimensions
    return tf.reduce_mean(mse)

class VAE_GAN_Model(keras.Model):
    def __init__(self, encoder, decoder, discriminator, steps_per_epoch=20,cycle_length=20, min_beta=0, max_beta=1,min_gamma=0, max_gamma=1, max_epochs = 100, **kwargs):
        super().__init__(**kwargs)
        self.max_epochs = max_epochs # Max number of epochs the model is going to be trained over

        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

        # per keras VAE example https://keras.io/examples/generative/vae/
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

        self.discriminator_loss_tracker = keras.metrics.Mean(name="discriminator_loss")
        self.gamma_tracker = keras.metrics.Mean(name="gamma")


        self.beta_tracker = keras.metrics.Mean(name="beta")
        self.steps_per_epoch = steps_per_epoch
        self.cycle_length = tf.cast(cycle_length, tf.float32)
        self.min_beta = tf.cast(min_beta, tf.float32)
        self.max_beta = tf.cast(max_beta, tf.float32)
        self.beta = tf.Variable(min_beta, dtype=tf.float32)

        self.min_gamma = tf.cast(min_gamma, tf.float32)
        self.max_gamma = tf.cast(max_gamma, tf.float32)
        self.gamma = tf.Variable(min_gamma, dtype=tf.float32)

    def compile(self, optimizer, **kwargs):
        super(VAE_GAN_Model, self).compile(**kwargs)
        # Set the optimizer for the entire model (encoder + decoder + discriminator)
        self.optimizer = optimizer

        # Collect trainable variables from encoder, decoder, and discriminator
        trainable_variables = (
            self.encoder.trainable_weights + 
            self.decoder.trainable_weights + 
            self.discriminator.trainable_weights
        )
        # Build the optimizer with the full variable list
        self.optimizer.build(trainable_variables)

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.discriminator_loss_tracker,
            self.beta_tracker,
        ]

    def cyclical_annealing_beta(self, epoch):
        cycle = tf.floor(1.0 + epoch / self.cycle_length)
        x = tf.abs(epoch / self.cycle_length - cycle + 1)
        # For first half (x < 0.5), scale 2x from 0 to 1
        # For second half (x >= 0.5), stay at 1
        scaled_x = tf.where(x < 0.5, 2.0 * x, 1.0)
        return self.min_beta + (self.max_beta - self.min_beta) * scaled_x
    
    # def cyclical_annealing_beta(self, epoch):
    #     # Get position within current cycle (0 to cycle_length)
    #     cycle_position = tf.math.mod(epoch, self.cycle_length)
        
    #     # For first half of cycle, increase linearly
    #     # For second half, stay at max
    #     half_cycle = self.cycle_length / 2
    #     scaled_x = tf.where(cycle_position <= half_cycle,
    #                     cycle_position / half_cycle,  # Linear increase in first half
    #                     1.0)                         # Stay at max for second half
        
    #     return self.min_beta + (self.max_beta - self.min_beta) * scaled_x


    # def get_gamma_schedule(self, epoch):
    #     # Convert to float32 for TF operations
    #     epoch = tf.cast(epoch, tf.float32)
        
    #     # Calculate annealing progress
    #     anneal_progress = (epoch - 50.0) / 50.0
    #     gamma_anneal = self.min_gamma + (self.max_gamma - self.min_gamma) * anneal_progress
        
    #     # Implement the conditions using tf.where
    #     gamma = tf.where(epoch < 50.0, 
    #                     0.0,  # if epoch < 50
    #                     tf.where(epoch >= 100.0,
    #                             self.max_gamma,  # if epoch >= 100
    #                             gamma_anneal))   # if 50 <= epoch < 100
        
    #     return gamma

    def get_gamma_schedule(self, epoch):
        # Convert to float32 for TF operations
        epoch = tf.cast(epoch, tf.float32)
        
        # Calculate annealing progress
        anneal_progress = (epoch - 0.0) / self.max_epochs
        gamma_anneal = self.min_gamma + (self.max_gamma - self.min_gamma) * anneal_progress
        # slope = (self.max_gamma -  self.min_gamma)/self.max_epochs

        return gamma_anneal


    def train_step(self, data):
        # Is this the beta tuning?
        epoch = tf.cast(self.optimizer.iterations / self.steps_per_epoch, tf.float32)
        
        # Update beta
        self.beta.assign(self.cyclical_annealing_beta(epoch))
        self.gamma.assign(self.get_gamma_schedule(epoch))

        # ---------------------------
        # Train the Discriminator
        # ---------------------------
        with tf.GradientTape() as tape_disc:
            # Generate reconstructed data
            mask = K.cast(K.not_equal(data, 0), K.floatx())
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # Get discriminator predictions
            real_output = self.discriminator(data)
            fake_output = self.discriminator(reconstruction * mask)
            
            # Labels for real and fake data
            real_labels = tf.ones_like(real_output)
            fake_labels = tf.zeros_like(fake_output)
            
            # Discriminator loss
            d_loss_real = keras.losses.binary_crossentropy(real_labels, real_output)
            d_loss_fake = keras.losses.binary_crossentropy(fake_labels, fake_output)
            d_loss = d_loss_real + d_loss_fake
            d_loss = tf.reduce_mean(d_loss)
        
        grads_disc = tape_disc.gradient(d_loss, self.discriminator.trainable_weights)
        self.optimizer.apply_gradients(zip(grads_disc, self.discriminator.trainable_weights))

        # ---------------------------
        # Train the VAE (Generator)
        # ---------------------------
        with tf.GradientTape() as tape:
            # Giving clean data here versus noisy in the atlas version
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # here we shove in our custom reconstructionn loss function
            # Ignore zero-padded entries. 
            mask = K.cast(K.not_equal(data, 0), K.floatx()) 
            reconstruction_loss = custom_mse_loss_with_multi_index_scaling(mask*reconstruction, mask*data)

            # This is just standard Kullback-Leibler diversion loss. I think this can stay.
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(kl_loss)

            # Generator (VAE) wants to fool the discriminator
            fake_output = self.discriminator(mask*reconstruction)
            valid_labels = tf.ones_like(fake_output)  # Try to make discriminator think reconstructions are real
            g_loss_adv = keras.losses.binary_crossentropy(valid_labels, fake_output)
            g_loss_adv = tf.reduce_mean(g_loss_adv)

            # curr_training_gamma = self.gamma * (epoch / 50)  # TODO 50 is arbitrary based on max_epochs # Not sure what this is doing.
            
            total_loss = reconstruction_loss*(1-self.beta) + kl_loss*self.beta + g_loss_adv * self.gamma

        # ----- Review for differences
        # grads = tape.gradient(total_loss, self.trainable_weights)
        # self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # self.total_loss_tracker.update_state(total_loss)
        # self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        # self.kl_loss_tracker.update_state(kl_loss)

        grads_vae = tape.gradient(total_loss, self.encoder.trainable_weights + self.decoder.trainable_weights)
        self.optimizer.apply_gradients(zip(grads_vae, self.encoder.trainable_weights + self.decoder.trainable_weights)) # This line is different
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.discriminator_loss_tracker.update_state(g_loss_adv)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reco_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
            "disc_loss": self.discriminator_loss_tracker.result(),
            "raw_loss": self.reconstruction_loss_tracker.result() + self.kl_loss_tracker.result(),
            "w_kl_loss": self.kl_loss_tracker.result() * self.beta,
            "w_disc_loss": self.discriminator_loss_tracker.result() * self.gamma,
            "beta": self.beta,
            "gamma": self.gamma,
        }
    
    # Since we overrode train_step we need test_step
    def test_step(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        mask = K.cast(K.not_equal(data, 0), K.floatx())
        reconstruction_loss = custom_mse_loss_with_multi_index_scaling(mask*data, mask*reconstruction)

        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(kl_loss)
        # Discriminator loss (only for monitoring)
        # pass both data and reconstruction through D to get generator adversarial loss
        real_output = self.discriminator(data)
        fake_output = self.discriminator(mask*reconstruction)
        real_labels = tf.ones_like(real_output)
        fake_labels = tf.zeros_like(fake_output)
        d_loss_real = keras.losses.binary_crossentropy(real_labels, real_output)
        d_loss_fake = keras.losses.binary_crossentropy(fake_labels, fake_output)
        d_loss = d_loss_real + d_loss_fake
        d_loss = tf.reduce_mean(d_loss)
        
        # Generator adversarial loss
        valid_labels = tf.ones_like(fake_output)
        g_loss_adv = keras.losses.binary_crossentropy(valid_labels, fake_output)
        g_loss_adv = tf.reduce_mean(g_loss_adv)
        total_loss = reconstruction_loss + kl_loss * self.beta + g_loss_adv * self.gamma
        
        return {
            "loss": total_loss,
            "reco_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "raw_loss": reconstruction_loss + kl_loss,
            "disc_loss": g_loss_adv,
            "w_kl_loss": kl_loss * self.beta,
            "w_disc_loss": g_loss_adv * self.gamma,
            "gamma": self.gamma,
            "beta" : self.beta
        }


    def call(self, data):
        z_mean,z_log_var,x = self.encoder(data)
        reconstruction = self.decoder(x)
        return {
            "z_mean": z_mean,
            "z_log_var": z_log_var,
            "reconstruction": reconstruction
        }

