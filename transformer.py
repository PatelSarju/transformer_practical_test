import tensorflow as tf


# ============================================================
# POSITIONAL ENCODING
# ============================================================

# Create positional encoding to provide word-position information
# because self-attention does not inherently understand word order.

def positional_encoding(sequence_length, d_model):

    # Create position numbers for each token in the sequence.
    position = tf.cast(
        tf.range(sequence_length)[:, tf.newaxis],
        tf.float32
    )

    # Create dimension numbers for each positional-encoding dimension.
    dimension = tf.cast(
        tf.range(d_model)[tf.newaxis, :],
        tf.float32
    )

    # Calculate different frequency rates for each pair of dimensions.
    angle_rates = 1 / tf.pow(
        10000.0,
        (2 * tf.floor(dimension / 2)) /
        tf.cast(d_model, tf.float32)
    )

    # Calculate the angle for every position and dimension.
    angle_radians = position * angle_rates

    # Identify even dimensions for applying sine.
    even_dimensions = tf.equal(
        tf.math.floormod(
            tf.range(d_model),
            2
        ),
        0
    )[tf.newaxis, :]

    # Apply sine to even dimensions and cosine to odd dimensions.
    positional_encoding = tf.where(
        even_dimensions,
        tf.sin(angle_radians),
        tf.cos(angle_radians)
    )

    # Add a batch dimension so the positional encoding can be
    # broadcast across all sequences in the batch.
    positional_encoding = positional_encoding[
        tf.newaxis, ...
    ]

    # Return the positional encoding.
    return positional_encoding


# ============================================================
# PADDING MASK
# ============================================================

# Create a padding mask so the attention mechanism does not
# attend to padding tokens.
#
# Token ID 0 is assumed to be the padding token.
#
# Input shape:
#     (batch_size, sequence_length)
#
# Output shape:
#     (batch_size, 1, 1, sequence_length)

def create_padding_mask(tokens):

    # Check which token positions contain the padding token 0.
    mask = tf.cast(
        tf.equal(tokens, 0),
        tf.float32
    )

    # Add dimensions so the mask can be broadcast
    # across attention heads and query positions.
    mask = mask[:, tf.newaxis, tf.newaxis, :]

    # Return the padding mask.
    return mask


# ============================================================
# MULTI-HEAD ATTENTION
# ============================================================

# Create Multi-Head Attention layer.

class MultiHeadAttention(tf.keras.layers.Layer):

    # Initialize the Multi-Head Attention layer.

    def __init__(self, d_model, num_heads):

        # Initialize the parent TensorFlow layer.
        super().__init__()

        # Store the total model dimension.
        self.d_model = d_model

        # Store the number of attention heads.
        self.num_heads = num_heads

        # Check that d_model can be evenly divided among heads.
        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads."
            )

        # Calculate the dimension handled by each attention head.
        self.depth = d_model // num_heads

        # Create a Dense layer to generate Query representations.
        self.query_layer = tf.keras.layers.Dense(
            d_model
        )

        # Create a Dense layer to generate Key representations.
        self.key_layer = tf.keras.layers.Dense(
            d_model
        )

        # Create a Dense layer to generate Value representations.
        self.value_layer = tf.keras.layers.Dense(
            d_model
        )

        # Create a Dense layer to combine the outputs of all attention heads.
        self.output_layer = tf.keras.layers.Dense(
            d_model
        )

    # Split the input representation into multiple attention heads.

    def split_heads(self, x, batch_size):

        # Reshape the representation so d_model is divided among
        # the specified number of attention heads.
        x = tf.reshape(
            x,
            (
                batch_size,
                -1,
                self.num_heads,
                self.depth
            )
        )

        # Move the attention-head dimension before the sequence dimension.
        x = tf.transpose(
            x,
            perm=[0, 2, 1, 3]
        )

        # Return the representation divided into attention heads.
        return x

    # Perform Multi-Head Self-Attention.

    def call(self, x, mask=None):

        # Get the batch size from the input tensor.
        batch_size = tf.shape(x)[0]

        # Generate Query representations from the input.
        Q = self.query_layer(x)

        # Generate Key representations from the input.
        K = self.key_layer(x)

        # Generate Value representations from the input.
        V = self.value_layer(x)

        # Split Query representations into multiple attention heads.
        Q = self.split_heads(
            Q,
            batch_size
        )

        # Split Key representations into multiple attention heads.
        K = self.split_heads(
            K,
            batch_size
        )

        # Split Value representations into multiple attention heads.
        V = self.split_heads(
            V,
            batch_size
        )

        # Calculate the similarity between each Query and Key.
        scores = tf.matmul(
            Q,
            K,
            transpose_b=True
        )

        # Convert the head dimension to float for scaling.
        depth = tf.cast(
            self.depth,
            tf.float32
        )

        # Scale the attention scores to keep their values stable.
        scaled_scores = (
            scores /
            tf.math.sqrt(depth)
        )

        # Apply the padding mask when it is provided.
        #
        # The padding mask marks padding tokens with 1.
        # Adding -1e9 makes their attention probability
        # approximately zero after softmax.
        if mask is not None:

            scaled_scores = (
                scaled_scores +
                (mask * -1e9)
            )

        # Convert the scaled scores into attention weights.
        attention_weights = tf.nn.softmax(
            scaled_scores,
            axis=-1
        )

        # Use the attention weights to calculate a weighted combination
        # of the Value representations.
        attention_output = tf.matmul(
            attention_weights,
            V
        )

        # Move the sequence dimension before the attention-head dimension.
        attention_output = tf.transpose(
            attention_output,
            perm=[0, 2, 1, 3]
        )

        # Combine all attention heads back into the original d_model dimension.
        attention_output = tf.reshape(
            attention_output,
            (
                batch_size,
                -1,
                self.d_model
            )
        )

        # Apply a final Dense layer to produce the attention output.
        output = self.output_layer(
            attention_output
        )

        # Return the final Multi-Head Attention output.
        return output


# ============================================================
# LOOK-AHEAD MASK
# ============================================================

# Create a look-ahead mask to prevent the decoder from
# attending to future tokens.

def create_look_ahead_mask(sequence_length):

    # Create a matrix containing ones above the main diagonal.
    mask = 1 - tf.linalg.band_part(
        tf.ones(
            (
                sequence_length,
                sequence_length
            )
        ),
        -1,
        0
    )

    # Add dimensions so the mask can work with
    # multi-head attention.
    mask = mask[
        tf.newaxis,
        tf.newaxis,
        :,
        :
    ]

    # Return the look-ahead mask.
    return mask


# ============================================================
# MASKED MULTI-HEAD ATTENTION
# ============================================================

# Create Masked Multi-Head Attention layer for the decoder.

class MaskedMultiHeadAttention(
    tf.keras.layers.Layer
):

    # Initialize the Masked Multi-Head Attention layer.

    def __init__(self, d_model, num_heads):

        # Initialize the parent TensorFlow layer.
        super().__init__()

        # Store the total model dimension.
        self.d_model = d_model

        # Store the number of attention heads.
        self.num_heads = num_heads

        # Check that d_model can be evenly divided among heads.
        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads."
            )

        # Calculate the dimension handled by each attention head.
        self.depth = d_model // num_heads

        # Create a Dense layer to generate Query representations.
        self.query_layer = tf.keras.layers.Dense(
            d_model
        )

        # Create a Dense layer to generate Key representations.
        self.key_layer = tf.keras.layers.Dense(
            d_model
        )

        # Create a Dense layer to generate Value representations.
        self.value_layer = tf.keras.layers.Dense(
            d_model
        )

        # Create a Dense layer to combine the outputs of all attention heads.
        self.output_layer = tf.keras.layers.Dense(
            d_model
        )

    # Split the input representation into multiple attention heads.

    def split_heads(self, x, batch_size):

        # Reshape the representation so d_model is divided among
        # the specified number of attention heads.
        x = tf.reshape(
            x,
            (
                batch_size,
                -1,
                self.num_heads,
                self.depth
            )
        )

        # Move the attention-head dimension before the sequence dimension.
        return tf.transpose(
            x,
            perm=[0, 2, 1, 3]
        )

    # Perform masked self-attention in the decoder.

    def call(self, x, *, mask=None):

        # Get the batch size from the input tensor.
        batch_size = tf.shape(x)[0]

        # Generate Query representations from the decoder input.
        Q = self.query_layer(x)

        # Generate Key representations from the decoder input.
        K = self.key_layer(x)

        # Generate Value representations from the decoder input.
        V = self.value_layer(x)

        # Split Query representations into multiple attention heads.
        Q = self.split_heads(
            Q,
            batch_size
        )

        # Split Key representations into multiple attention heads.
        K = self.split_heads(
            K,
            batch_size
        )

        # Split Value representations into multiple attention heads.
        V = self.split_heads(
            V,
            batch_size
        )

        # Calculate the similarity between each Query and Key.
        scores = tf.matmul(
            Q,
            K,
            transpose_b=True
        )

        # Convert the head dimension to float for scaling.
        depth = tf.cast(
            self.depth,
            tf.float32
        )

        # Scale the attention scores to keep their values stable.
        scaled_scores = (
            scores /
            tf.math.sqrt(depth)
        )

        # Apply the combined look-ahead and padding mask.
        if mask is not None:

            scaled_scores = (
                scaled_scores +
                (mask * -1e9)
            )

        # Convert the masked scores into attention weights.
        attention_weights = tf.nn.softmax(
            scaled_scores,
            axis=-1
        )

        # Use the attention weights to calculate a weighted combination
        # of the Value representations.
        attention_output = tf.matmul(
            attention_weights,
            V
        )

        # Move the sequence dimension before the attention-head dimension.
        attention_output = tf.transpose(
            attention_output,
            perm=[0, 2, 1, 3]
        )

        # Combine all attention heads back into the original d_model dimension.
        attention_output = tf.reshape(
            attention_output,
            (
                batch_size,
                -1,
                self.d_model
            )
        )

        # Apply a final Dense layer to produce the attention output.
        output = self.output_layer(
            attention_output
        )

        # Return the final masked attention output.
        return output


# ============================================================
# CROSS ATTENTION
# ============================================================

# Create Cross-Attention layer to connect the Decoder with the Encoder.

class CrossAttention(
    tf.keras.layers.Layer
):

    # Initialize the Cross-Attention layer.

    def __init__(self, d_model, num_heads):

        # Initialize the parent TensorFlow layer.
        super().__init__()

        # Store the total model dimension.
        self.d_model = d_model

        # Store the number of attention heads.
        self.num_heads = num_heads

        # Check that d_model can be evenly divided among heads.
        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads."
            )

        # Calculate the dimension handled by each attention head.
        self.depth = d_model // num_heads

        # Create a Dense layer to generate Query representations
        # from the Decoder output.
        self.query_layer = tf.keras.layers.Dense(
            d_model
        )

        # Create a Dense layer to generate Key representations
        # from the Encoder output.
        self.key_layer = tf.keras.layers.Dense(
            d_model
        )

        # Create a Dense layer to generate Value representations
        # from the Encoder output.
        self.value_layer = tf.keras.layers.Dense(
            d_model
        )

        # Create a Dense layer to combine the outputs of all attention heads.
        self.output_layer = tf.keras.layers.Dense(
            d_model
        )

    # Split the input representation into multiple attention heads.

    def split_heads(self, x, batch_size):

        # Reshape the representation so d_model is divided among
        # the specified number of attention heads.
        x = tf.reshape(
            x,
            (
                batch_size,
                -1,
                self.num_heads,
                self.depth
            )
        )

        # Move the attention-head dimension before the sequence dimension.
        return tf.transpose(
            x,
            perm=[0, 2, 1, 3]
        )

    # Perform Cross-Attention between the Decoder and Encoder outputs.

    def call(
        self,
        decoder_input,
        encoder_output,
        mask=None
    ):

        # Get the batch size from the Decoder input.
        batch_size = tf.shape(
            decoder_input
        )[0]

        # Generate Query representations from the Decoder.
        Q = self.query_layer(
            decoder_input
        )

        # Generate Key representations from the Encoder output.
        K = self.key_layer(
            encoder_output
        )

        # Generate Value representations from the Encoder output.
        V = self.value_layer(
            encoder_output
        )

        # Split Query representations into multiple attention heads.
        Q = self.split_heads(
            Q,
            batch_size
        )

        # Split Key representations into multiple attention heads.
        K = self.split_heads(
            K,
            batch_size
        )

        # Split Value representations into multiple attention heads.
        V = self.split_heads(
            V,
            batch_size
        )

        # Calculate the similarity between Decoder Queries
        # and Encoder Keys.
        scores = tf.matmul(
            Q,
            K,
            transpose_b=True
        )

        # Convert the head dimension to float for scaling.
        depth = tf.cast(
            self.depth,
            tf.float32
        )

        # Scale the attention scores to keep their values stable.
        scaled_scores = (
            scores /
            tf.math.sqrt(depth)
        )

        # Apply the Encoder padding mask.
        if mask is not None:

            scaled_scores = (
                scaled_scores +
                (mask * -1e9)
            )

        # Convert the scaled scores into attention weights.
        attention_weights = tf.nn.softmax(
            scaled_scores,
            axis=-1
        )

        # Use the attention weights to calculate a weighted combination
        # of the Encoder Value representations.
        attention_output = tf.matmul(
            attention_weights,
            V
        )

        # Move the sequence dimension before the attention-head dimension.
        attention_output = tf.transpose(
            attention_output,
            perm=[0, 2, 1, 3]
        )

        # Combine all attention heads back into the original d_model dimension.
        attention_output = tf.reshape(
            attention_output,
            (
                batch_size,
                -1,
                self.d_model
            )
        )

        # Apply a final Dense layer to produce the attention output.
        output = self.output_layer(
            attention_output
        )

        # Return the final Cross-Attention output.
        return output


# ============================================================
# ENCODER
# ============================================================

# Create the Encoder block of the Transformer.

class Encoder(tf.keras.layers.Layer):

    # Initialize the Encoder.

    def __init__(
        self,
        paragraph_vocab_size,
        d_model,
        num_heads,
        dff
    ):

        # Initialize the parent TensorFlow layer.
        super().__init__()

        # Store the total model dimension.
        self.d_model = d_model

        # Convert paragraph token IDs into d_model-dimensional embeddings.
        self.embedding = tf.keras.layers.Embedding(
            paragraph_vocab_size,
            d_model
        )

        # Create Multi-Head Self-Attention for the Encoder.
        self.attention = MultiHeadAttention(
            d_model,
            num_heads
        )

        # Create the first Layer Normalization.
        self.layer_norm_1 = (
            tf.keras.layers.LayerNormalization()
        )

        # Create the Feed-Forward Network.
        self.feed_forward = tf.keras.Sequential([

            # Expand each token representation to the dff dimension.
            tf.keras.layers.Dense(
                dff,
                activation="relu"
            ),

            # Convert the representation back to d_model dimensions.
            tf.keras.layers.Dense(
                d_model
            )
        ])

        # Create the second Layer Normalization.
        self.layer_norm_2 = (
            tf.keras.layers.LayerNormalization()
        )

    # Define the processing workflow of the Encoder.

    def call(
        self,
        tokens,
        padding_mask=None
    ):

        # Get the length of the input paragraph token sequence.
        sequence_length = tf.shape(tokens)[1]

        # Convert paragraph token IDs into dense vector representations.
        x = self.embedding(tokens)

        # Create positional encoding for the input paragraph.
        position = positional_encoding(
            sequence_length,
            self.d_model
        )

        # Add positional information to the token embeddings.
        x = x + position

        # Apply Multi-Head Self-Attention.
        #
        # The padding mask prevents attention from using
        # padded positions.
        attention_output = self.attention(
            x,
            mask=padding_mask
        )

        # Add the attention output to the original representation
        # using a residual connection.
        x = x + attention_output

        # Normalize the result of the residual connection.
        x = self.layer_norm_1(x)

        # Apply the Feed-Forward Network independently to each position.
        feed_forward_output = self.feed_forward(x)

        # Add the Feed-Forward output to the previous representation
        # using a residual connection.
        x = x + feed_forward_output

        # Normalize the result of the second residual connection.
        x = self.layer_norm_2(x)

        # Return the final Encoder representation.
        return x


# ============================================================
# DECODER
# ============================================================

# Create the Decoder block of the Transformer.

class Decoder(tf.keras.layers.Layer):

    # Initialize the Decoder.

    def __init__(
        self,
        summary_vocab_size,
        d_model,
        num_heads,
        dff
    ):

        # Initialize the parent TensorFlow layer.
        super().__init__()

        # Store the total model dimension.
        self.d_model = d_model

        # Convert summary token IDs into d_model-dimensional embeddings.
        self.embedding = tf.keras.layers.Embedding(
            summary_vocab_size,
            d_model
        )

        # Create Masked Multi-Head Self-Attention for the Decoder.
        self.masked_attention = (
            MaskedMultiHeadAttention(
                d_model,
                num_heads
            )
        )

        # Create the first Layer Normalization.
        self.layer_norm_1 = (
            tf.keras.layers.LayerNormalization()
        )

        # Create Cross-Attention to connect the Decoder
        # with the Encoder output.
        self.cross_attention = CrossAttention(
            d_model,
            num_heads
        )

        # Create the second Layer Normalization.
        self.layer_norm_2 = (
            tf.keras.layers.LayerNormalization()
        )

        # Create the Feed-Forward Network.
        self.feed_forward = tf.keras.Sequential([

            # Expand each token representation to the dff dimension.
            tf.keras.layers.Dense(
                dff,
                activation="relu"
            ),

            # Convert the representation back to d_model dimensions.
            tf.keras.layers.Dense(
                d_model
            )
        ])

        # Create the third Layer Normalization.
        self.layer_norm_3 = (
            tf.keras.layers.LayerNormalization()
        )

    # Define the processing workflow of the Decoder.

    def call(
        self,
        tokens,
        encoder_output,
        padding_mask=None
    ):

        # Get the length of the Decoder token sequence.
        sequence_length = tf.shape(tokens)[1]

        # Convert Decoder token IDs into dense vector representations.
        x = self.embedding(tokens)

        # Create positional encoding for the Decoder sequence.
        position = positional_encoding(
            sequence_length,
            self.d_model
        )

        # Add positional information to the Decoder embeddings.
        x = x + position

        # Create a look-ahead mask to prevent the Decoder
        # from accessing future summary tokens.
        look_ahead_mask = create_look_ahead_mask(
            sequence_length
        )

        # Create a padding mask for the Decoder tokens.
        decoder_padding_mask = create_padding_mask(
            tokens
        )

        # Combine the look-ahead mask and padding mask.
        #
        # This ensures that the Decoder:
        # 1. Cannot see future tokens.
        # 2. Cannot attend to padding tokens.
        combined_mask = tf.maximum(
            look_ahead_mask,
            decoder_padding_mask
        )

        # Apply Masked Self-Attention.
        masked_attention_output = (
            self.masked_attention(
                x,
                mask=combined_mask
            )
        )

        # Add the masked attention output using a residual connection.
        x = x + masked_attention_output

        # Normalize the result of the first residual connection.
        x = self.layer_norm_1(x)

        # Apply Cross-Attention so the Decoder can use
        # information from the paragraph Encoder output.
        #
        # The encoder padding mask prevents the Decoder
        # from attending to padded article positions.
        cross_attention_output = (
            self.cross_attention(
                x,
                encoder_output,
                mask=padding_mask
            )
        )

        # Add the Cross-Attention output using a residual connection.
        x = x + cross_attention_output

        # Normalize the result of the second residual connection.
        x = self.layer_norm_2(x)

        # Apply the Feed-Forward Network to each Decoder position.
        feed_forward_output = (
            self.feed_forward(x)
        )

        # Add the Feed-Forward output using a residual connection.
        x = x + feed_forward_output

        # Normalize the result of the third residual connection.
        x = self.layer_norm_3(x)

        # Return the final Decoder representation.
        return x


# ============================================================
# COMPLETE TRANSFORMER
# ============================================================

# Create the complete Encoder-Decoder Transformer model.

class Transformer(tf.keras.Model):

    # Initialize the complete Transformer.

    def __init__(
        self,
        paragraph_vocab_size,
        summary_vocab_size,
        d_model,
        num_heads,
        dff
    ):

        # Initialize the parent TensorFlow model.
        super().__init__()

        # Create the Encoder using the paragraph vocabulary size.
        self.encoder = Encoder(
            paragraph_vocab_size,
            d_model,
            num_heads,
            dff
        )

        # Create the Decoder using the summary vocabulary size.
        self.decoder = Decoder(
            summary_vocab_size,
            d_model,
            num_heads,
            dff
        )

        # Create the final Dense layer to produce a score
        # for every token in the summary vocabulary.
        self.output_layer = tf.keras.layers.Dense(
            summary_vocab_size
        )

    # Define the complete Encoder-Decoder workflow.

    def call(
        self,
        paragraph_tokens,
        decoder_tokens
    ):

        # Create a padding mask for the paragraph.
        paragraph_padding_mask = create_padding_mask(
            paragraph_tokens
        )

        # Pass the paragraph token IDs through the Encoder
        # to create contextual representations.
        encoder_output = self.encoder(
            paragraph_tokens,
            padding_mask=paragraph_padding_mask
        )

        # Pass the Decoder input and Encoder output through
        # the Decoder to generate summary representations.
        decoder_output = self.decoder(
            decoder_tokens,
            encoder_output,
            padding_mask=paragraph_padding_mask
        )

        # Convert the Decoder representations into logits
        # for every token in the summary vocabulary.
        logits = self.output_layer(
            decoder_output
        )

        # Return the logits for the predicted summary tokens.
        return logits
    
    
    
    
