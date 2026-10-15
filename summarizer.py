import tensorflow as tf
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
from transformer import Transformer

# CONFIGURATION

# CSV file containing article and summary pairs.
CSV_FILE = "news_article.csv"

# Maximum number of tokens allowed in an article.
MAX_ARTICLE_LENGTH = 256

# Maximum number of tokens allowed in a summary.
MAX_SUMMARY_LENGTH = 64

# Number of samples processed at one time.
BATCH_SIZE = 8

# Number of times the complete training dataset is processed.
EPOCHS = 100

# Transformer model dimension.
D_MODEL = 128

# Number of attention heads.
NUM_HEADS = 4

# Feed-Forward Network dimension.
DFF = 512

# Learning rate for Adam optimizer.
LEARNING_RATE = 0.001


# LOAD DATASET

# Load the article-summary dataset from the CSV file.
data = pd.read_csv(CSV_FILE)

# Remove rows where article or summary is missing.
data = data.dropna(
    subset=["article", "summary"]
)

# Convert article and summary columns to strings.
data["article"] = data["article"].astype(str)
data["summary"] = data["summary"].astype(str)

# Get all articles from the dataset.
paragraphs = data["article"].tolist()

# Get all summaries from the dataset.
summaries = data["summary"].tolist()


# ADD START AND END TOKENS

# Add <start> and <end> tokens to every summary.
summaries = [
    "<start> " + summary + " <end>"
    for summary in summaries
]


# CREATE ARTICLE TOKENIZER

# Create a tokenizer for articles.
#
# <unk> is used when the tokenizer encounters a word
# that was not present in the training vocabulary.
paragraph_tokenizer = (
    tf.keras.preprocessing.text.Tokenizer(
        oov_token="<unk>"
    )
)

# Build the article vocabulary.
paragraph_tokenizer.fit_on_texts(
    paragraphs
)

# Convert all articles into token IDs.
paragraph_tokens = (
    paragraph_tokenizer.texts_to_sequences(
        paragraphs
    )
)

# CREATE SUMMARY TOKENIZER

# Create a tokenizer for summaries.
#
# filters="" is important because we want to preserve
# the <start> and <end> tokens.
summary_tokenizer = (
    tf.keras.preprocessing.text.Tokenizer(
        filters="",
        oov_token="<unk>"
    )
)

# Build the summary vocabulary.
summary_tokenizer.fit_on_texts(
    summaries
)

# Convert all summaries into token IDs.
summary_tokens = (
    summary_tokenizer.texts_to_sequences(
        summaries
    )
)

# LIMIT ARTICLE LENGTH

# Store the final article token sequences.
processed_paragraph_tokens = []

# Process every article.
for tokens in paragraph_tokens:

    # Keep only the first MAX_ARTICLE_LENGTH tokens.
    tokens = tokens[
        :MAX_ARTICLE_LENGTH
    ]

    # Store the processed article.
    processed_paragraph_tokens.append(
        tokens
    )

# CREATE DECODER INPUTS AND TARGETS

# Store decoder input sequences.
decoder_inputs = []

# Store target sequences.
targets = []


# Process every summary.
for tokens in summary_tokens:

    # We want the final decoder sequence to contain
    # at most MAX_SUMMARY_LENGTH tokens.

    # Keep one extra token because decoder input
    # removes <end> and target removes <start>.
    tokens = tokens[
        :MAX_SUMMARY_LENGTH + 1
    ]

    # Decoder receives everything except <end>.
    decoder_input = tokens[:-1]

    # Target contains everything except <start>.
    target = tokens[1:]

    # Store decoder input.
    decoder_inputs.append(
        decoder_input
    )

    # Store target.
    targets.append(
        target
    )

# CREATE TF.DATA DATASET

# Create a generator that returns one training example
# at a time.
def data_generator():

    # Go through every training example.
    for article, decoder_input, target in zip(
        processed_paragraph_tokens,
        decoder_inputs,
        targets
    ):

        # Return article, decoder input and target.
        yield (
            article,
            decoder_input,
            target
        )


# Create a TensorFlow Dataset from the generator.
dataset = tf.data.Dataset.from_generator(
    data_generator,

    # Specify the data types returned by the generator.
    output_signature=(

        tf.TensorSpec(
            shape=(None,),
            dtype=tf.int32
        ),

        tf.TensorSpec(
            shape=(None,),
            dtype=tf.int32
        ),

        tf.TensorSpec(
            shape=(None,),
            dtype=tf.int32
        )
    )
)


# Shuffle the training examples.
dataset = dataset.shuffle(
    buffer_size=len(processed_paragraph_tokens)
)


# CREATE BATCHES

# Create batches and pad sequences inside each batch.
#
# Article:
#     maximum length = 256
#
# Decoder input:
#     maximum length = 64
#
# Target:
#     maximum length = 64
dataset = dataset.padded_batch(
    BATCH_SIZE,

    # Specify the final size of every sequence.
    padded_shapes=(

        [MAX_ARTICLE_LENGTH],

        [MAX_SUMMARY_LENGTH],

        [MAX_SUMMARY_LENGTH]
    ),

    # Use 0 as the padding token.
    padding_values=(

        tf.constant(
            0,
            dtype=tf.int32
        ),

        tf.constant(
            0,
            dtype=tf.int32
        ),

        tf.constant(
            0,
            dtype=tf.int32
        )
    ),

    # Keep the last incomplete batch.
    drop_remainder=False
)


# Improve input pipeline performance.
dataset = dataset.prefetch(
    tf.data.AUTOTUNE
)


# GET VOCABULARY SIZES

# Tokenizer word indexes start from 1.
# Token ID 0 is reserved for padding.
#
# Therefore, vocabulary size is:
#
# maximum token ID + 1

paragraph_vocab_size = (
    len(
        paragraph_tokenizer.word_index
    ) + 1
)

summary_vocab_size = (
    len(
        summary_tokenizer.word_index
    ) + 1
)


# CREATE TRANSFORMER

# Create the Transformer model.
transformer = Transformer(
    paragraph_vocab_size=paragraph_vocab_size,
    summary_vocab_size=summary_vocab_size,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dff=DFF
)

# CREATE LOSS FUNCTION

# Create Sparse Categorical Cross-Entropy loss.
#
# reduction="none" gives a separate loss value
# for every token.
loss_function = (
    tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction="none"
    )
)

# CALCULATE MASKED LOSS

# Calculate loss while ignoring padding tokens.
def calculate_loss(real, predicted):

    # Calculate loss for every token.
    loss = loss_function(
        real,
        predicted
    )

    # Create a mask.
    #
    # Real token:
    #     1
    #
    # Padding token:
    #     0
    mask = tf.cast(
        tf.not_equal(
            real,
            0
        ),
        tf.float32
    )

    # Remove the loss contribution from padding tokens.
    loss = loss * mask

    # Calculate the total loss of real tokens.
    total_loss = tf.reduce_sum(
        loss
    )

    # Count the number of real tokens.
    total_tokens = tf.reduce_sum(
        mask
    )

    # Return average loss.
    return (
        total_loss /
        tf.maximum(
            total_tokens,
            1.0
        )
    )

# CREATE OPTIMIZER

# Create Adam optimizer.
optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE
)


# TRAINING STEP

# Create one training step.
@tf.function
def train_step(
    paragraph_batch,
    decoder_input_batch,
    target_batch
):

    # Record operations so TensorFlow can calculate gradients.
    with tf.GradientTape() as tape:

        # Pass the article and decoder input
        # through the Transformer.
        logits = transformer(
            paragraph_batch,
            decoder_input_batch
        )

        # Calculate the loss.
        loss = calculate_loss(
            target_batch,
            logits
        )

    # Calculate gradients of the loss
    # with respect to Transformer parameters.
    gradients = tape.gradient(
        loss,
        transformer.trainable_variables
    )

    # Update Transformer parameters.
    optimizer.apply_gradients(
        zip(
            gradients,
            transformer.trainable_variables
        )
    )

    # Return the loss.
    return loss


# TRAIN THE TRANSFORMER

# Repeat training for the specified number of epochs.
for epoch in range(EPOCHS):

    # Store the losses from all batches.
    epoch_losses = []

    # Process one batch at a time.
    for (
        paragraph_batch,
        decoder_input_batch,
        target_batch
    ) in dataset:

        # Perform one training step.
        loss = train_step(
            paragraph_batch,
            decoder_input_batch,
            target_batch
        )

        # Store the batch loss.
        epoch_losses.append(
            loss
        )

    # Calculate the average loss for this epoch.
    average_loss = tf.reduce_mean(
        epoch_losses
    )

    # Print progress every 5 epochs.
    if (epoch + 1) % 10 == 0:

        print(
            f"Epoch {epoch + 1}/{EPOCHS}, "
            f"Loss: {average_loss.numpy():.4f}"
        )


# TAKE USER ARTICLE

# Ask the user for an article.
user_paragraph = input(
    "\nEnter a paragraph: "
)


# TOKENIZE USER ARTICLE

# Convert the user's article into token IDs.
user_tokens = (
    paragraph_tokenizer.texts_to_sequences(
        [user_paragraph]
    )
)


# Keep only the first MAX_ARTICLE_LENGTH tokens.
user_tokens = user_tokens[
    0
][:MAX_ARTICLE_LENGTH]


# Convert the article into a TensorFlow tensor.
user_input = tf.constant(
    [user_tokens],
    dtype=tf.int32
)


# GET START AND END TOKENS

# Get the token ID of <start>.
start_token = (
    summary_tokenizer
    .word_index["<start>"]
)

# Get the token ID of <end>.
end_token = (
    summary_tokenizer
    .word_index["<end>"]
)


# START SUMMARY GENERATION

# Start the summary with <start>.
generated_tokens = [
    start_token
]


# GENERATE SUMMARY

# Generate one token at a time.
for _ in range(MAX_SUMMARY_LENGTH - 1):

    # Convert generated tokens into a TensorFlow tensor.
    decoder_input = tf.constant(
        [generated_tokens],
        dtype=tf.int32
    )

    # Pass the article and currently generated
    # summary tokens through the Transformer.
    logits = transformer(
        user_input,
        decoder_input
    )

    # Select the logits of the last generated position.
    next_token_logits = (
        logits[:, -1, :]
    )

    # Select the token having the highest logit.
    next_token = tf.argmax(
        next_token_logits,
        axis=-1
    ).numpy()[0]

    # Convert NumPy integer into normal Python integer.
    next_token = int(
        next_token
    )

    # Add the predicted token to the generated sequence.
    generated_tokens.append(
        next_token
    )

    # Stop generation when <end> is predicted.
    if next_token == end_token:

        break


# CONVERT TOKEN IDS INTO WORDS

# Convert token IDs back into words.
generated_words = [
    summary_tokenizer.index_word[token]
    for token in generated_tokens
    if token in summary_tokenizer.index_word
]

# REMOVE START AND END TOKENS

# Remove <start> from the beginning.
if (
    generated_words
    and generated_words[0] == "<start>"
):

    generated_words = (
        generated_words[1:]
    )


# Remove <end> from the end.
if (
    generated_words
    and generated_words[-1] == "<end>"
):

    generated_words = (
        generated_words[:-1]
    )

# CREATE FINAL SUMMARY

# Convert the generated words into one sentence.
summary = " ".join(
    generated_words
)


# PRINT GENERATED SUMMARY

print(
    "\nGenerated summary:",
    summary
)


# CALCULATE BLEU SCORE

# Ask the user to enter the correct/reference summary.
reference_summary = input(
    "\nEnter the correct summary: "
)

# Convert the reference summary into individual words.
reference_words = (
    reference_summary.split()
)

# Convert the generated summary into individual words.
generated_words = (
    summary.split()
)


# Calculate BLEU score.
bleu_score = corpus_bleu(
    [[reference_words]],
    [generated_words]
)

# Print BLEU score.
print(
    f"BLEU Score: {bleu_score:.4f}"
)




