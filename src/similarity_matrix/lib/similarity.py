import os
# Avoid VRAM fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import gc
import numpy as np

from similarity_matrix.lib.model import encode, cos_sim
from similarity_matrix.lib.model import get_model_size, get_memory_capacity
from similarity_matrix.lib.logging import logger


# -----------------------------------------------------------------------
# Memory safe cosine similarity computation

def cos_sim_mem(model,
                row_texts: list[str],
                column_texts: list[str],
                max_memory_gb: float = None) -> np.ndarray:
    """
    Compute cosine similarity matrix between row and column texts with automatic memory management.

    Args:
        model: The embedding model
        row_texts: List of texts for rows
        column_texts: List of texts for columns
        max_memory_gb: Maximum memory to use in GB (default: None, auto-detect)

    Returns:
        np.ndarray: Similarity matrix of shape (len(row_texts), len(column_texts))
    """

    if max_memory_gb is None:
        max_memory_gb = get_memory_capacity() * 0.8  # Use 80% of available memory

    # Get current GPU memory status
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        total_memory = torch.cuda.get_device_properties(0).total_memory
        allocated_memory = torch.cuda.memory_allocated(0)
        reserved_memory = torch.cuda.memory_reserved(0)
        free_memory = total_memory - reserved_memory

        logger.info("GPU Memory Status:")
        logger.info(f"  Total: {total_memory / (1024**3):.2f} GB")
        logger.info(f"  Allocated: {allocated_memory / (1024**3):.2f} GB")
        logger.info(f"  Reserved: {reserved_memory / (1024**3):.2f} GB")
        logger.info(f"  Free: {free_memory / (1024**3):.2f} GB")

    # Estimate model size and calculate optimal batch sizes
    model_size_bytes = get_model_size(model)
    model_size_gb = model_size_bytes / (1024**3)

    logger.info(f"Model size: {model_size_gb:.2f} GB")

    # Estimate embedding dimension with a very small test
    test_embedding = encode(model, ["test"])
    embedding_dim = test_embedding.shape[1]
    del test_embedding
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(f"Embedding dimension: {embedding_dim}")

    # Calculate memory requirements per embedding (float32)
    bytes_per_embedding = embedding_dim * 4

    # Use actual available GPU memory if CUDA is available, otherwise use
    # max_memory_gb
    if torch.cuda.is_available():
        # Use only 70% of free memory to be conservative
        available_memory_bytes = int(free_memory * 0.7)
        logger.info(
            f"Using 70% of free GPU memory: {available_memory_bytes / (1024**3):.2f} GB")
    else:
        # Use 80% of specified limit
        available_memory_bytes = int(max_memory_gb * 0.8 * (1024**3))

    # Ensure minimum available memory
    min_memory_bytes = 256 * 1024**2  # 256MB minimum
    available_memory_bytes = max(available_memory_bytes, min_memory_bytes)

    # Conservative batch size calculation
    # Account for: input tensors, attention matrices, gradients, intermediate results
    # Use safety factor of 6x for transformer model memory overhead
    safety_factor = 6
    max_sequence_length = 512  # Typical max sequence length for embedding models

    # Memory per item includes: input embeddings + attention overhead
    memory_per_item = bytes_per_embedding * safety_factor

    # Calculate initial conservative batch size
    initial_batch_size = max(1, available_memory_bytes // memory_per_item)

    # Start with very conservative batch sizes and adjust based on data size
    max_row_batch = min(initial_batch_size, len(row_texts))
    max_col_batch = min(initial_batch_size, len(column_texts))

    # Further reduce batch size for large datasets
    if len(row_texts) > 100000 or len(column_texts) > 100000:
        max_row_batch = min(max_row_batch, 1000)
        max_col_batch = min(max_col_batch, 1000)

    # Even more conservative for very large datasets
    if len(row_texts) > 500000 or len(column_texts) > 500000:
        max_row_batch = min(max_row_batch, 500)
        max_col_batch = min(max_col_batch, 500)

    # Ensure minimum batch size of 1
    max_row_batch = max(1, max_row_batch)
    max_col_batch = max(1, max_col_batch)

    logger.info(
        f"Initial batch sizes: rows={max_row_batch}, columns={max_col_batch}")
    logger.info(
        f"Dataset sizes: rows={len(row_texts)}, " +
        f"columns={len(column_texts)}")
    logger.info(
        f"Available memory for processing: {available_memory_bytes / (1024**3):.2f} GB")

    # Initialize result matrix
    similarity_matrix = np.zeros(
        (len(row_texts), len(column_texts)), dtype=float)

    # Process in batches with adaptive batch size reduction
    batch_reduction_factor = 1
    max_retries = 3

    for row_start in range(0, len(row_texts), max_row_batch):
        row_end = min(row_start + max_row_batch, len(row_texts))

        # Adaptive batch size for rows
        current_row_batch_size = max_row_batch // batch_reduction_factor
        current_row_batch_size = max(1, current_row_batch_size)

        # Adjust row_end based on current batch size
        if row_start + current_row_batch_size < row_end:
            row_end = row_start + current_row_batch_size

        row_batch = row_texts[row_start:row_end]

        logger.info(
            f"Processing row batch {row_start}:{row_end} (" +
            f"{len(row_batch)} items)")

        # Generate embeddings for current row batch with retry logic
        row_embeddings = None
        retry_count = 0

        while row_embeddings is None and retry_count < max_retries:
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Further reduce batch size if we've had failures
                if retry_count > 0:
                    reduced_batch_size = len(row_batch) // (2 ** retry_count)
                    reduced_batch_size = max(1, reduced_batch_size)
                    if reduced_batch_size < len(row_batch):
                        logger.warning(
                            f"Reducing row batch size to {reduced_batch_size} due to memory constraints")
                        row_batch = row_batch[:reduced_batch_size]
                        row_end = row_start + reduced_batch_size

                row_embeddings = encode(model, row_batch)
                logger.debug(
                    f"Successfully encoded {len(row_batch)} row items")

            except torch.cuda.OutOfMemoryError as e:
                retry_count += 1
                logger.warning(
                    f"CUDA OOM on row batch (attempt {retry_count}/{max_retries}): {e}")

                # Check if the row_embeddings variable was already initialized
                # and clear it to free memory
                if 'row_embeddings' in locals():
                    del row_embeddings

                # Call garbage collector to free up memory
                gc.collect()

                # Clear CUDA cache to free up memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if retry_count >= max_retries:
                    logger.error(
                        f"Failed to process row batch after {max_retries} attempts")
                    raise

                # Wait a moment for memory to clear
                import time
                time.sleep(1)

                # Let the local variable name available
                row_embeddings = None

        # Process column batches
        for col_start in range(0, len(column_texts), max_col_batch):
            col_end = min(col_start + max_col_batch, len(column_texts))

            # Adaptive batch size for columns
            current_col_batch_size = max_col_batch // batch_reduction_factor
            current_col_batch_size = max(1, current_col_batch_size)

            # Adjust col_end based on current batch size
            if col_start + current_col_batch_size < col_end:
                col_end = col_start + current_col_batch_size

            col_batch = column_texts[col_start:col_end]

            logger.debug(
                f"Processing column batch {col_start}:{col_end} " +
                f"({len(col_batch)} items)")

            # Generate embeddings for current column batch with retry logic
            column_embeddings = None
            retry_count = 0

            while column_embeddings is None and retry_count < max_retries:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Further reduce batch size if we've had failures
                    if retry_count > 0:
                        reduced_batch_size = len(
                            col_batch) // (2 ** retry_count)
                        reduced_batch_size = max(1, reduced_batch_size)
                        if reduced_batch_size < len(col_batch):
                            logger.warning(
                                f"Reducing column batch size to {reduced_batch_size} due to memory constraints")
                            col_batch = col_batch[:reduced_batch_size]
                            col_end = col_start + reduced_batch_size

                    column_embeddings = encode(model, col_batch)
                    logger.debug(
                        f"Successfully encoded {len(col_batch)} column items")

                except torch.cuda.OutOfMemoryError as e:
                    retry_count += 1
                    logger.warning(
                        f"CUDA OOM on column batch (attempt {retry_count}/{max_retries}): {e}")

                    # Check if the column_embeddings variable was already initialized
                    # and clear it to free memory
                    if 'column_embeddings' in locals():
                        del column_embeddings

                    # Call garbage collector to free up memory
                    gc.collect()

                    # Clear CUDA cache to free up memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if retry_count >= max_retries:
                        logger.error(
                            f"Failed to process column batch after {max_retries} attempts")
                        raise

                    # Wait a moment for memory to clear
                    import time
                    time.sleep(1)

                    # Let the local variable name available
                    column_embeddings = None

            # Calculate cosine similarity for this batch with retry logic
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                similarity_tensor = cos_sim(row_embeddings, column_embeddings)

                # Store results in the appropriate section of the result matrix
                similarity_matrix[row_start:row_end, col_start:col_end] = similarity_tensor.cpu(
                ).numpy().astype(float)

                logger.debug(
                    f"Computed similarity for batch [{row_start}:{row_end}, {col_start}:{col_end}]")

            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA OOM during similarity computation: {e}")
                # Try to free up memory and retry with smaller batches
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # If similarity computation fails, we need to increase batch
                # reduction
                batch_reduction_factor *= 2
                logger.warning(
                    f"Increasing batch reduction factor to {batch_reduction_factor}")
                raise  # Re-raise to restart with smaller batches

            # Clear column embeddings to free memory
            del column_embeddings
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Clear row embeddings to free memory
        del row_embeddings
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Log memory status periodically
        if torch.cuda.is_available() and (row_start // max_row_batch) % 10 == 0:
            current_allocated = torch.cuda.memory_allocated(0)
            current_reserved = torch.cuda.memory_reserved(0)
            logger.info(
                "Memory status after batch: Allocated=" +
                f"{current_allocated / (1024**3):.2f}GB, " +
                f"Reserved={current_reserved / (1024**3):.2f}GB")

    logger.info("Completed similarity matrix computation")
    return similarity_matrix
