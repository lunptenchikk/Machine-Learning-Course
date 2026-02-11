# Diemid Rybchenko, SI 2025/2026
import os
import csv
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def make_out_dirs(mode: str):
    
    base = os.path.join("out", mode)
    
    
    samples = os.path.join(base, "samples")
    
    
    os.makedirs(samples, exist_ok=True)
    return base, samples

def one_hot(labels, num_classes=10):
    
    return tf.one_hot(tf.cast(labels, tf.int32), depth=num_classes)

def tile_labels_as_channels(y_onehot, h=28, w=28):
    y = y_onehot[:, None, None, :]
    y = tf.tile(y, [1, h, w, 1])
    return y

def save_grid_images(generator, epoch, samples_dir, latent_dim=128, num_classes=10):
    
    
    n_rows, n_cols = num_classes, 10
    z = tf.random.normal([n_rows * n_cols, latent_dim])
    

    labels = np.repeat(np.arange(num_classes), n_cols)
    
    y = one_hot(labels, num_classes)

    fake = generator([z, y], training=False)
    fake = (fake + 1.0) / 2.0  

    
    fake = tf.reshape(fake, [n_rows, n_cols, 28, 28, 1])

    fake = tf.transpose(fake, [0, 2, 1, 3, 4])  
    
    fake = tf.reshape(fake, [n_rows * 28, n_cols * 28, 1])

    img = tf.clip_by_value(fake * 255.0, 0, 255)
    
    img = tf.cast(img, tf.uint8).numpy()

    path = os.path.join(samples_dir, f"epoch_{epoch:03d}.png")
    keras.utils.save_img(path, img, scale=False)


def load_dataset(name: str, batch_size: int):
    if name == "mnist":
        
        (x_train, y_train), _ = keras.datasets.mnist.load_data()
        
    elif name == "fashion_mnist":
        (x_train, y_train), _ = keras.datasets.fashion_mnist.load_data()
        
    else:
        raise ValueError("dataset must be mnist or fashion_mnist")

    x_train = x_train.astype("float32")
    
    
    x_train = (x_train / 127.5) - 1.0 
    x_train = np.expand_dims(x_train, axis=-1)  

    ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    
    ds = ds.shuffle(60000).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds


def build_generator(latent_dim=128, num_classes=10):
    z_in = layers.Input(shape=(latent_dim,))
    
    y_in = layers.Input(shape=(num_classes,))
    x = layers.Concatenate()([z_in, y_in])

    x = layers.Dense(7 * 7 * 128, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.ReLU()(x)
    x = layers.Reshape((7, 7, 128))(x)

    x = layers.Conv2DTranspose(128, 4, strides=2, padding="same", use_bias=False)(x)
    
    
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    out = layers.Conv2D(1, 3, padding="same", activation="tanh")(x)
    return keras.Model([z_in, y_in], out, name="G")

def build_discriminator(num_classes=10):
    
    
    x_in = layers.Input(shape=(28, 28, 1))
    y_in = layers.Input(shape=(num_classes,))

    y_map = layers.Lambda(lambda t: tile_labels_as_channels(t, 28, 28))(y_in)
    
    x = layers.Concatenate(axis=-1)([x_in, y_map])  

    x = layers.Conv2D(64, 4, strides=2, padding="same")(x)
    
    
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)
    

    x = layers.Conv2D(128, 4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model([x_in, y_in], out, name="D")

def build_critic(num_classes=10):
    
    x_in = layers.Input(shape=(28, 28, 1))
    y_in = layers.Input(shape=(num_classes,))

    y_map = layers.Lambda(lambda t: tile_labels_as_channels(t, 28, 28))(y_in)
    
    x = layers.Concatenate(axis=-1)([x_in, y_map])  

    x = layers.Conv2D(64, 4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(128, 4, strides=2, padding="same")(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Flatten()(x)
    
    
    out = layers.Dense(1)(x)  # linear
    return keras.Model([x_in, y_in], out, name="C")

bce = keras.losses.BinaryCrossentropy(from_logits=False)

def dcgan_losses(d_real, d_fake):
    
    
    d_loss = bce(tf.ones_like(d_real), d_real) + bce(tf.zeros_like(d_fake), d_fake)
    
    g_loss = bce(tf.ones_like(d_fake), d_fake)
    return d_loss, g_loss

def wgan_losses(c_real, c_fake):
    
    
    c_loss = tf.reduce_mean(c_fake) - tf.reduce_mean(c_real)
    
    
    g_loss = -tf.reduce_mean(c_fake)
    
    
    
    wasserstein_est = tf.reduce_mean(c_real) - tf.reduce_mean(c_fake)
    return c_loss, g_loss, wasserstein_est

def gradient_penalty(critic, real, fake, y_onehot):
    
    
    batch = tf.shape(real)[0]
    alpha = tf.random.uniform([batch, 1, 1, 1], 0.0, 1.0)
    
    x_hat = alpha * real + (1.0 - alpha) * fake

    with tf.GradientTape() as tape:
        
        tape.watch(x_hat)
        c_hat = critic([x_hat, y_onehot], training=True)
    grads = tape.gradient(c_hat, x_hat)
    
    grads = tf.reshape(grads, [batch, -1])
    norm = tf.norm(grads, axis=1)
    
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

@tf.function
def train_step_dcgan(generator, discriminator, g_opt, d_opt, real_x, y_onehot, latent_dim):
    z = tf.random.normal([tf.shape(real_x)[0], latent_dim])

    with tf.GradientTape() as d_tape:
        fake_x = generator([z, y_onehot], training=True)
        
        
        d_real = discriminator([real_x, y_onehot], training=True)
        d_fake = discriminator([fake_x, y_onehot], training=True)
        
        d_loss, g_loss_dummy = dcgan_losses(d_real, d_fake)

    d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
    d_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))

    with tf.GradientTape() as g_tape:
        
        fake_x = generator([z, y_onehot], training=True)
        
        d_fake = discriminator([fake_x, y_onehot], training=True)
        _, g_loss = dcgan_losses(tf.zeros_like(d_fake), d_fake)
        

    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
    
    g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))

    return d_loss, g_loss, tf.constant(0.0), tf.constant(0.0)

@tf.function
def train_step_wgan(generator, critic, g_opt, c_opt, real_x, y_onehot, latent_dim, clip_value):
    z = tf.random.normal([tf.shape(real_x)[0], latent_dim])

    with tf.GradientTape() as c_tape:
        fake_x = generator([z, y_onehot], training=True)
        
        c_real = critic([real_x, y_onehot], training=True)
        
        c_fake = critic([fake_x, y_onehot], training=True)
        c_loss, g_loss_dummy, w_est = wgan_losses(c_real, c_fake)

    c_grads = c_tape.gradient(c_loss, critic.trainable_variables)
    
    c_opt.apply_gradients(zip(c_grads, critic.trainable_variables))

    # weight clipping
    for v in critic.trainable_variables:
        
        v.assign(tf.clip_by_value(v, -clip_value, clip_value))

    # generator step
    with tf.GradientTape() as g_tape:
        
        fake_x = generator([z, y_onehot], training=True)
        c_fake = critic([fake_x, y_onehot], training=True)
        g_loss = -tf.reduce_mean(c_fake)

    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
    
    g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))

    return c_loss, g_loss, w_est, tf.constant(0.0)

@tf.function
def train_step_wgangp(generator, critic, g_opt, c_opt, real_x, y_onehot, latent_dim, gp_lambda):
    
    
    z = tf.random.normal([tf.shape(real_x)[0], latent_dim])

    with tf.GradientTape() as c_tape:
        fake_x = generator([z, y_onehot], training=True)
        c_real = critic([real_x, y_onehot], training=True)
        
        
        
        c_fake = critic([fake_x, y_onehot], training=True)
        c_loss, g_loss_dummy, w_est = wgan_losses(c_real, c_fake)
        
        
        gp = gradient_penalty(critic, real_x, fake_x, y_onehot)
        c_loss = c_loss + gp_lambda * gp

    c_grads = c_tape.gradient(c_loss, critic.trainable_variables)
    
    c_opt.apply_gradients(zip(c_grads, critic.trainable_variables))

    with tf.GradientTape() as g_tape:
        fake_x = generator([z, y_onehot], training=True)
        
        
        c_fake = critic([fake_x, y_onehot], training=True)
        g_loss = -tf.reduce_mean(c_fake)

    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
    
    
    g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))

    return c_loss, g_loss, w_est, gp

def main():
    ap = argparse.ArgumentParser()
    
    ap.add_argument("--mode", type=str, default="dcgan", choices=["dcgan", "wgan", "wgangp"])
    
    ap.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashion_mnist"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    
    
    ap.add_argument("--latent", type=int, default=128)
    ap.add_argument("--n_critic", type=int, default=5)
    ap.add_argument("--clip", type=float, default=0.01)
    
    ap.add_argument("--gp_lambda", type=float, default=10.0)
    args = ap.parse_args()

    base_dir, samples_dir = make_out_dirs(args.mode)
    
    metrics_path = os.path.join(base_dir, "metrics.csv")

    ds = load_dataset(args.dataset, args.batch)
    G = build_generator(args.latent)

    if args.mode == "dcgan":
        
        
        D = build_discriminator()
        g_opt = keras.optimizers.Adam(2e-4, beta_1=0.5)
        d_opt = keras.optimizers.Adam(2e-4, beta_1=0.5)
        
        
        model_second = D
    else:
        C = build_critic()
        if args.mode == "wgan":
            
            g_opt = keras.optimizers.RMSprop(5e-5)
            
            c_opt = keras.optimizers.RMSprop(5e-5)
        else:
            
            
            g_opt = keras.optimizers.Adam(1e-4, beta_1=0.0, beta_2=0.9)
            c_opt = keras.optimizers.Adam(1e-4, beta_1=0.0, beta_2=0.9)
        model_second = C

    # metrics header
    if not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "step", "c_or_d_loss", "g_loss", "wasserstein_est", "gp"])

    global_step = 0
    
    for epoch in range(args.epochs):
        for real_x, y in ds:
            y_oh = one_hot(y, 10)
            
            

            if args.mode == "dcgan":
                d_loss, g_loss, w_est, gp = train_step_dcgan(G, model_second, g_opt, d_opt, real_x, y_oh, args.latent)
                
                

            elif args.mode == "wgan":
                c_loss_acc = 0.0
                w_est_acc = 0.0
                
                
                for _ in range(args.n_critic):
                    c_loss, g_loss, w_est, gp = train_step_wgan(G, model_second, g_opt, c_opt, real_x, y_oh, args.latent, args.clip)
                    c_loss_acc += c_loss
                    w_est_acc += w_est
                    
                    
                d_loss, w_est = c_loss_acc / args.n_critic, w_est_acc / args.n_critic

            else:  # wgangp
                c_loss_acc = 0.0
                w_est_acc = 0.0
                gp_acc = 0.0
                for _ in range(args.n_critic):
                    c_loss, g_loss, w_est, gp = train_step_wgangp(G, model_second, g_opt, c_opt, real_x, y_oh, args.latent, args.gp_lambda)
                    c_loss_acc += c_loss
                    w_est_acc += w_est
                    
                    
                    gp_acc += gp
                d_loss, w_est, gp = c_loss_acc / args.n_critic, w_est_acc / args.n_critic, gp_acc / args.n_critic





            with open(metrics_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                
                
                w.writerow([epoch, global_step, float(d_loss), float(g_loss), float(w_est), float(gp)])
            global_step += 1

        save_grid_images(G, epoch, samples_dir, args.latent, 10)
        print(f"[{args.mode}] epoch {epoch+1}/{args.epochs}: loss={float(d_loss):.4f} g={float(g_loss):.4f} w={float(w_est):.4f} gp={float(gp):.4f}")

if __name__ == "__main__":
    main()
