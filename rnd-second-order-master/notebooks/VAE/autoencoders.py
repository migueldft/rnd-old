from tensorflow.keras import backend as k
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, BatchNormalization, AlphaDropout
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GaussianNoise


def make_encoder(dropout=0.2):
    return Sequential(
        [
            Dense(256, activation="selu"),
            BatchNormalization(),
            Dense(128, activation="selu"),
            AlphaDropout(dropout),
            Dense(64, activation="selu"),
            AlphaDropout(dropout),
            Dense(32, activation="selu"),
        ]
    )


def make_decoder(output_dim, dropout=0.2):
    return Sequential(
        [
            Dense(64, activation="selu"),
            BatchNormalization(),
            Dense(128, activation="selu"),
            AlphaDropout(dropout),
            Dense(256, activation="selu"),
            AlphaDropout(dropout),
            Dense(output_dim, activation="sigmoid"),
        ]
    )


def make_conditional_autoencoder(mu_encoder, log_sigma_encoder, decoder, input_shape, optimizer):
    x = Input(shape=(input_shape,))
    y = Input(shape=(1,))
    mu = mu_encoder(x)
    log_sigma = log_sigma_encoder(x)
    x_rec = decoder(GaussianNoise(k.exp(log_sigma))(mu))
    model = Model(inputs=[x, y], outputs=x_rec)

    def reconstruction_loss(x_true, x_rec):
        return k.mean(k.binary_crossentropy(x_true, x_rec), axis=1)

    def representation_loss(x_true, x_rec):
        base = k.mean(-log_sigma + k.exp(2.0 * log_sigma) - 0.5, axis=1)
        mu_term = y * k.mean(mu ** 2, axis=1) + (1 - y) * k.mean((mu - 5.0) ** 2, axis=1)
        return base + mu_term

    def total_loss(x_true, x_rec):
        return reconstruction_loss(x_true, x_rec) + representation_loss(x_true, x_rec)

    model.compile(optimizer=optimizer, loss=total_loss, metrics=[reconstruction_loss, representation_loss])
    return model

