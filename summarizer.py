import tensorflow as tf
import pandas as pd
from nltk.translate.bleu_score import corpus_bleu
from transformer import Transformer


# TRAINING DATA

# Load the paragraph-summary dataset.
data = pd.read_csv(
    "paragraph_summary_dataset.csv"
)

# Create a dictionary containing paragraphs
# and their corresponding summaries.
summary_data = dict(
    zip(
        data["paragraph"],
        data["summary"]
    )
)

# SEPARATE PARAGRAPHS AND SUMMARIES

# Get all paragraphs from the dictionary.
paragraphs = list(
    summary_data.keys()
)

# Add <start> and <end> tokens to every summary.
summaries = [
    "<start> " + summary + " <end>"
    for summary in summary_data.values()
]


# CREATE PARAGRAPH TOKENIZER

# Create tokenizer for paragraphs.
paragraph_tokenizer = (
    tf.keras.preprocessing.text.Tokenizer()
)

# Build paragraph vocabulary.
paragraph_tokenizer.fit_on_texts(
    paragraphs
)

# Convert all paragraphs into token IDs.
paragraph_tokens = (
    paragraph_tokenizer
    .texts_to_sequences(paragraphs)
)


# CREATE SUMMARY TOKENIZER

# Create summary tokenizer.
# filters="" keeps <start> and <end>.
summary_tokenizer = (
    tf.keras.preprocessing.text.Tokenizer(
        filters=""
    )
)

# Build summary vocabulary.
summary_tokenizer.fit_on_texts(
    summaries
)

# Convert all summaries into token IDs.
summary_tokens = (
    summary_tokenizer
    .texts_to_sequences(summaries)
)


# CREATE DECODER INPUT AND TARGET

# Store decoder inputs for all summaries.
decoder_inputs = []

# Store target summaries.
targets = []


# Process every summary.
for tokens in summary_tokens:

    # Decoder receives everything except <end>.
    decoder_input = tokens[:-1]

    # Target contains everything except <start>.
    target = tokens[1:]

    # Add decoder input to the list.
    decoder_inputs.append(
        decoder_input
    )

    # Add target to the list.
    targets.append(
        target
    )


# PAD THE SEQUENCES

# Pad paragraphs so they have the same length.
paragraph_tokens = (
    tf.keras.preprocessing.sequence
    .pad_sequences(
        paragraph_tokens,
        padding="post"
    )
)

# Pad decoder inputs so they have the same length.
decoder_inputs = (
    tf.keras.preprocessing.sequence
    .pad_sequences(
        decoder_inputs,
        padding="post"
    )
)

# Pad target summaries so they have the same length.
targets = (
    tf.keras.preprocessing.sequence
    .pad_sequences(
        targets,
        padding="post"
    )
)

# CONVERT INPUTS INTO TENSORS

# Convert paragraph token IDs into TensorFlow tensor.
paragraph_input = tf.constant(
    paragraph_tokens,
    dtype=tf.int32
)

# Convert decoder input IDs into TensorFlow tensor.
decoder_input_tensor = tf.constant(
    decoder_inputs,
    dtype=tf.int32
)

# Convert target IDs into TensorFlow tensor.
target_tensor = tf.constant(
    targets,
    dtype=tf.int32
)


# GET VOCABULARY SIZES

# Get paragraph vocabulary size.
paragraph_vocab_size = (
    max(
        paragraph_tokenizer.word_index.values()
    ) + 1
)

# Get summary vocabulary size.
summary_vocab_size = (
    max(
        summary_tokenizer.word_index.values()
    ) + 1
)


# CREATE TRANSFORMER

# Create the Transformer model.
transformer = Transformer(
    paragraph_vocab_size=paragraph_vocab_size,
    summary_vocab_size=summary_vocab_size,
    d_model=128,
    num_heads=2,
    dff=512
)


# CREATE LOSS FUNCTION

# Create Sparse Categorical Cross-Entropy loss.
loss_function = (
    tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction="none"
    )
)


# Calculate loss while ignoring padding tokens.
def calculate_loss(real, predicted):

    # Calculate loss for every token.
    loss = loss_function(
        real,
        predicted
    )

    # Create mask for non-padding tokens.
    mask = tf.cast(
        tf.not_equal(real, 0),
        tf.float32
    )

    # Ignore padding token loss.
    loss = loss * mask

    # Calculate average loss.
    return (
        tf.reduce_sum(loss) /
        tf.reduce_sum(mask)
    )


# CREATE OPTIMIZER

# Create Adam optimizer.
optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.001
)


# TRAIN THE TRANSFORMER

# Number of training iterations.
epochs = 350

# Repeat training multiple times.
for epoch in range(epochs):

    # Record operations for gradient calculation.
    with tf.GradientTape() as tape:

        # Pass all paragraphs and decoder inputs
        # through the Transformer.
        logits = transformer(
            paragraph_input,
            decoder_input_tensor
        )

        # Calculate loss for all summaries.
        loss = calculate_loss(
            target_tensor,
            logits
        )

    # Calculate gradients.
    gradients = tape.gradient(
        loss,
        transformer.trainable_variables
    )

    # Update Transformer weights.
    optimizer.apply_gradients(
        zip(
            gradients,
            transformer.trainable_variables
        )
    )

    # Print loss every 50 epochs.
    if (epoch + 1) % 50 == 0:

        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {loss.numpy():.4f}"
        )


# TAKE USER PARAGRAPH

# Ask the user for a paragraph.
user_paragraph = input(
    "\nEnter a paragraph: "
)


# TOKENIZE USER PARAGRAPH


# Convert the user's paragraph into token IDs.
user_tokens = (
    paragraph_tokenizer
    .texts_to_sequences(
        [user_paragraph]
    )
)

# Convert token IDs into TensorFlow tensor.
user_input = tf.constant(
    user_tokens,
    dtype=tf.int32
)


# GET START AND END TOKEN

# Get <start> token ID.
start_token = (
    summary_tokenizer
    .word_index["<start>"]
)

# Get <end> token ID.
end_token = (
    summary_tokenizer
    .word_index["<end>"]
)


# START SUMMARY GENERATION

# Start generation with <start>.
generated_tokens = [
    start_token
]


# GENERATE SUMMARY TOKENS

# Generate summary tokens one by one.
for _ in range(30):

    # Convert generated tokens into a tensor.
    decoder_input = tf.constant(
        [generated_tokens],
        dtype=tf.int32
    )

    # Pass the paragraph and generated
    # summary tokens through the Transformer.
    logits = transformer(
        user_input,
        decoder_input
    )

    # Take prediction from the last position.
    next_token_logits = (
        logits[:, -1, :]
    )

    # Select the token with the highest probability.
    next_token = tf.argmax(
        next_token_logits,
        axis=-1
    ).numpy()[0]

    # Add predicted token to the summary.
    generated_tokens.append(
        int(next_token)
    )

    # Stop when <end> is generated.
    if next_token == end_token:
        break


# CONVERT TOKEN IDs INTO WORDS

# Convert generated token IDs into words.
generated_words = [
    summary_tokenizer.index_word[token]
    for token in generated_tokens
    if token in summary_tokenizer.index_word
]


# CREATE FINAL SUMMARY

# Remove <start> and <end> from the final output.
generated_words = generated_words[1:-1]

# Create final summary.
summary = " ".join(
    generated_words
)


# PRINT SUMMARY

print(
    "\nGenerated summary:",
    summary
)


# CALCULATE BLEU SCORE

# Ask for the correct summary.
reference_summary = input(
    "Enter the correct summary: "
)

# Convert reference summary into words.
reference_words = (
    reference_summary.split()
)

# Convert generated summary into words.
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


