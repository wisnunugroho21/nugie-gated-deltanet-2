import math
from typing import Any

import flax.nnx as nnx
import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from model import HybridGDN2LM


# --- Wrap Hugging Face Dataset in a Grain Data Source ---
class HuggingFaceDataSource(grain.RandomAccessDataSource):
    """
    A Grain wrapper for Hugging Face datasets.
    Because HF relies on Apache Arrow under the hood, random lookups are incredibly fast.
    """

    def __init__(self, hf_ds: Dataset) -> None:
        self.hf_ds = hf_ds

    def __len__(self) -> int:
        return len(self.hf_ds)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # HF natively returns a dictionary for the row (e.g., {'text': 'Some string'})
        return self.hf_ds[index]


class TokenizerAndShift(grain.MapTransform):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 128) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def map(self, element: dict[str, Any]) -> dict[str, Any]:
        encoded: dict[str, list[int]] = self.tokenizer(
            element["text"],
            truncation=True,
            max_length=self.max_length + 1,  # +1 for the shift
            padding="max_length",
            return_tensors=None,
        )

        tokens: list[int] = encoded["input_ids"]

        new_element = {
            "inputs": tokens[:-1],
            "targets": tokens[1:],
            "attention_mask": encoded["attention_mask"][:-1],
        }

        return new_element


class ConvertToJaxArrays(grain.MapTransform):
    def map(self, element: dict[str, Any]) -> dict[str, Any]:
        for key in ["inputs", "targets", "attention_mask"]:
            element[key] = jnp.array(np.array(element[key]))
        return element


class FilterEmptyLines(grain.FilterTransform):
    def filter(self, element: dict[str, Any]) -> bool:
        return len(element["text"].strip()) > 0


def build_dataloader(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 8,
    max_length: int = 128,
) -> grain.DataLoader:
    source = HuggingFaceDataSource(dataset)

    sampler = grain.IndexSampler(
        num_records=len(source),
        num_epochs=1,
        shard_options=grain.ShardOptions(
            shard_index=0, shard_count=1, drop_remainder=True
        ),
        shuffle=True,
        seed=42,
    )

    loader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[
            FilterEmptyLines(),
            TokenizerAndShift(tokenizer, max_length=max_length),
            ConvertToJaxArrays(),
            grain.Batch(batch_size=batch_size, drop_remainder=True),
        ],
        worker_count=0,
    )

    return loader


def loss_fn(model: nnx.Module, batch: dict[str, jax.Array]) -> jax.Array:
    logits = model(batch["inputs"])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["targets"]
    ).mean()

    return loss


@nnx.jit
def train_step(
    model: nnx.Module, optimizer: nnx.Optimizer, batch: dict[str, jax.Array]
) -> jax.Array:
    grad_fn = nnx.value_and_grad(loss_fn)
    loss, grads = grad_fn(model, batch)

    optimizer.update(model, grads)
    return loss


@nnx.jit
def eval_step(model: nnx.Module, batch: dict[str, jax.Array]) -> jax.Array:
    logits = model(batch["inputs"])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["targets"]
    ).mean()

    return loss


@nnx.jit
def predict_next_token(
    model: nnx.Module, input_ids: jax.Array, last_idx: jax.Array
) -> jax.Array:
    """Runs a single forward pass to predict the next word.

    `input_ids` may be right-padded so its length is a multiple of the model's
    chunk_size; `last_idx` is the position of the last *real* token, from which
    the next-token logits are read. The model is causal, so right-padding after
    `last_idx` cannot influence the prediction at `last_idx`.
    """

    # 1. Get the raw scores (logits) for every token in the vocabulary
    logits = model(input_ids)

    # 2. Isolate the predictions for the last real token in our sequence
    # Shape goes from (batch, seq_len, vocab_size) -> (vocab_size,)
    next_token_logits = logits[0, last_idx, :]

    # 3. Greedy Decoding: Simply pick the token with the highest probability score
    next_token = jnp.argmax(next_token_logits)

    return next_token


def interactive_chat(
    model: HybridGDN2LM, tokenizer: PreTrainedTokenizer, max_new_tokens: int = 100
):
    """Starts an infinite loop for user interaction."""

    print("\n" + "=" * 50)
    print("🤖 Massive LLM is online and ready!")
    print("Type 'quit' or 'exit' to shut down the server.")
    print("=" * 50 + "\n")

    # Model constraints the generation loop must respect:
    #   • the GDN-2 kernel requires seq_len % chunk_size == 0
    #   • RoPE is only precomputed up to max_seq_len positions
    chunk_size: int = model.cells[0].gdn2.chunk_size
    max_seq_len: int = int(model.rope_cos[...].shape[0])

    # The Infinite Loop
    while True:
        # 1. Get User Input
        user_text = input("You: ")

        # 2. Check for exit commands
        if user_text.strip().lower() in ["quit", "exit"]:
            print("Shutting down the model. Goodbye!")
            break

        # Skip empty inputs
        if not user_text.strip():
            continue

        # 3. Tokenize the input into a standard NumPy array
        # We add batch dimension manually so shape is (1, seq_len)
        encoded = tokenizer(user_text, return_tensors="np")
        input_ids = encoded["input_ids"]

        print("Model: ", end="", flush=True)

        # 4. The Autoregressive Generation Loop
        for _ in range(max_new_tokens):
            # A. Slide the context to the most recent `max_seq_len` tokens so we
            #    never exceed the precomputed RoPE table.
            context = input_ids[:, -max_seq_len:]
            real_len = context.shape[1]

            # B. Right-pad to a multiple of chunk_size for the GDN-2 kernel.
            #    The last real token sits at index real_len - 1.
            pad_len = (-real_len) % chunk_size
            if pad_len:
                context = np.pad(context, ((0, 0), (0, pad_len)))

            # C. Predict the next token from the last real position
            next_token_array = predict_next_token(
                model, jnp.asarray(context), jnp.asarray(real_len - 1)
            )

            # Convert the JAX array back to a standard Python integer
            next_token_id = next_token_array.item()

            # B. Check for the End-Of-Sequence (EOS) token
            # If the model decides it is done talking, break the generation loop
            if next_token_id == tokenizer.eos_token_id:
                break

            # C. Decode the single integer back into a readable word
            word = tokenizer.decode([next_token_id])

            # Print the word immediately (flush=True forces the terminal to update)
            print(word, end="", flush=True)

            # D. Append the new token to our sequence so the model can read it
            # on the next loop iteration.
            input_ids = np.append(input_ids, [[next_token_id]], axis=1)

        # Print a newline when the model finishes its complete thought
        print("\n")


def train_and_evaluate(
    num_epochs: int = 1000, eval_every_n_steps: int = 5, save_every_n_steps: int = 5
):
    train_dataset: Dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="train"
    )
    val_dataset: Dataset = load_dataset(
        "Salesforce/wikitext", "wikitext-2-raw-v1", split="validation"
    )

    gpt2tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("gpt2")
    gpt2tokenizer.pad_token = gpt2tokenizer.eos_token

    train_loader: grain.DataLoader = build_dataloader(
        train_dataset, gpt2tokenizer, batch_size=8, max_length=128
    )
    val_loader: grain.DataLoader = build_dataloader(
        val_dataset, gpt2tokenizer, batch_size=8, max_length=128
    )

    # Initialize Model and Optimizer
    rngs: nnx.Rngs = nnx.Rngs(0)
    model: nnx.Module = HybridGDN2LM(
        vocab_size=gpt2tokenizer.vocab_size,  # must match the tokenizer's id range
        dim=128,
        num_heads=4,
        num_cells=2,
        mlp_dim=256,
        chunk_size=4,
        conv_kernel=4,
        window_size=16,
        max_seq_len=128,  # must be >= the dataloader's max_length (128)
        rope_theta=10_000.0,
        tie_embeddings=True,
        rngs=rngs,
    )

    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate=3e-4), wrt=nnx.Param)

    step = 0  # restore_checkpoint(mngr, model, optimizer, rngs, data_iterator)

    print("Starting training...")
    if step > 0:
        print(f"Resuming training from step {step}...")

    for epoch in range(num_epochs):
        # Re-create the iterator each epoch: the Grain sampler is configured
        # with num_epochs=1, so a single iterator is exhausted after one pass.
        data_iterator = iter(train_loader)

        total_train_loss = 0.0
        train_steps = 0

        for train_batch in data_iterator:
            train_loss = train_step(model, optimizer, train_batch)
            total_train_loss += train_loss
            train_steps += 1

        avg_train_loss = total_train_loss / train_steps
        perplexity = math.exp(avg_train_loss)

        print(
            f"Step {step:04d} | Train Loss: {avg_train_loss:.4f} | Perplexity: {perplexity:.2f} | Epoch: {epoch + 1}/{num_epochs}"
        )

        if step % eval_every_n_steps == 0 and step > 0:
            total_val_loss = 0.0
            val_steps = 0

            for val_batch in val_loader:
                val_loss = eval_step(model, val_batch)
                total_val_loss += val_loss
                val_steps += 1

            avg_val_loss = total_val_loss / val_steps
            perplexity = math.exp(avg_val_loss)

            print(
                f"Val Loss: {avg_val_loss:.4f} | Perplexity: {perplexity:.2f} | Epoch: {epoch + 1}/{num_epochs}"
            )

        step += 1

    step = 0  # Reset step count after each epoch

    interactive_chat(model, gpt2tokenizer, max_new_tokens=150)


if __name__ == "__main__":
    train_and_evaluate()
