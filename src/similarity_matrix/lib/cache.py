import pickle
import hashlib
from functools import wraps
from pathlib import Path

from similarity_matrix.lib.logging import logger


# -----------------------------------------------------------------------
# Caching

def file_cache(cache_dir="cache"):
    """
    Decorator that caches function results to disk using pickle files.

    Args:
        cache_dir (str): Directory to store cache files (default: "cache")
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache directory if it doesn't exist
            Path(cache_dir).mkdir(exist_ok=True)

            # Create a hash of the function name, args, and kwargs
            func_signature = {
                'function': func.__name__,
                'args': args,
                'kwargs': kwargs
            }

            # Convert to string and hash
            signature_str = str(func_signature)
            hash_obj = hashlib.md5(signature_str.encode())
            cache_filename = f"{hash_obj.hexdigest()}.pkl"
            cache_path = Path(cache_dir) / cache_filename

            # Check if cache file exists
            if cache_path.exists():
                try:
                    # Load from cache
                    with open(cache_path, 'rb') as f:
                        result = pickle.load(f)
                    logger.info(
                        f"Cache hit: Loaded result from {cache_filename} (func: " +
                        func.__name__ + ")")
                    return result
                except (pickle.PickleError, EOFError) as e:
                    logger.error(f"Cache file corrupted, removing: {e}")
                    cache_path.unlink()  # Remove corrupted file

            # Cache miss - execute function
            logger.debug(f"Cache miss: Computing result for {func.__name__}")
            result = func(*args, **kwargs)

            # Save result to cache
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(result, f)
                logger.info(f"Result cached to {cache_filename} (func: " +
                            func.__name__ + ")")
            except (pickle.PickleError, AttributeError, TypeError, RecursionError) as e:
                logger.warning(
                    f"Failed to cache result (object not pickleable): {e}")
                # Clean up the failed cache file
                if cache_path.exists():
                    try:
                        cache_path.unlink()
                    except OSError:
                        logger.warning(
                            f"Failed to clean up corrupted cache file: {cache_path}")
            except OSError as e:
                logger.error(f"Failed to write cache file: {e}")
                # Clean up the failed cache file
                if cache_path.exists():
                    try:
                        cache_path.unlink()
                    except OSError:
                        logger.warning(
                            f"Failed to clean up corrupted cache file: {cache_path}")

            return result

        # Add cache management methods
        def clear_cache():
            """Clear all cache files for this function"""
            cache_path = Path(cache_dir)
            if cache_path.exists():
                for file in cache_path.glob("*.pkl"):
                    file.unlink()
                logger.info(f"Cache cleared for {func.__name__}")

        def cache_info():
            """Get information about cached files"""
            cache_path = Path(cache_dir)
            if cache_path.exists():
                cache_files = list(cache_path.glob("*.pkl"))
                return {
                    'cache_dir': str(cache_path),
                    'cached_results': len(cache_files),
                    'total_size_bytes': sum(
                        f.stat().st_size for f in cache_files)}
            return {
                'cache_dir': str(cache_path),
                'cached_results': 0,
                'total_size_bytes': 0}

        wrapper.clear_cache = clear_cache
        wrapper.cache_info = cache_info

        return wrapper
    return decorator


# Example usage:

# if __name__ == "__main__":
#     import time

#     @file_cache(cache_dir="my_cache")
#     def expensive_computation(n, multiplier=2):
#         """Simulate an expensive computation"""
#         print(f"Computing {n} * {multiplier}...")
#         time.sleep(2)  # Simulate slow computation
#         return n * multiplier

#     @file_cache()  # Uses default cache directory
#     def fibonacci(n):
#         """Calculate fibonacci number (recursive for demonstration)"""
#         if n <= 1:
#             return n
#         return fibonacci(n-1) + fibonacci(n-2)

#     # Test the cache
#     print("=== Testing expensive_computation ===")
#     result1 = expensive_computation(10, multiplier=3)  # Cache miss
#     print(f"Result: {result1}\n")

#     result2 = expensive_computation(10, multiplier=3)  # Cache hit
#     print(f"Result: {result2}\n")

#     result3 = expensive_computation(10, multiplier=5)  # Different args, cache miss
#     print(f"Result: {result3}\n")

#     # Cache info
#     print("Cache info:", expensive_computation.cache_info())

#     # Clear cache
#     expensive_computation.clear_cache()
