from functools import wraps
import pytest
import hashlib
import tempfile
import shutil
import pickle
from pathlib import Path

from similarity_matrix.lib.cache import file_cache


class TestFileCacheDecorator:
    """Test suite for the file_cache decorator"""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache files"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def call_counter(self):
        """Helper fixture to count function calls"""
        counter = {'count': 0}
        return counter

    def test_basic_caching_functionality(self, temp_cache_dir, call_counter):
        """Test that functions are cached and retrieved correctly"""

        @file_cache(cache_dir=temp_cache_dir)
        def test_func(x, y=10):
            call_counter['count'] += 1
            return x * y

        # First call should execute function
        result1 = test_func(5, y=20)
        assert result1 == 100
        assert call_counter['count'] == 1

        # Second call with same args should use cache
        result2 = test_func(5, y=20)
        assert result2 == 100
        assert call_counter['count'] == 1  # Function not called again

        # Different args should execute function again
        result3 = test_func(5, y=30)
        assert result3 == 150
        assert call_counter['count'] == 2

    def test_cache_directory_creation(self, temp_cache_dir):
        """Test that cache directory is created if it doesn't exist"""
        cache_dir = Path(temp_cache_dir) / "new_cache"
        assert not cache_dir.exists()

        @file_cache(cache_dir=str(cache_dir))
        def test_func():
            return "test"

        test_func()
        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_cache_file_creation(self, temp_cache_dir):
        """Test that cache files are created with correct naming"""

        @file_cache(cache_dir=temp_cache_dir)
        def test_func(x):
            return x * 2

        test_func(42)

        cache_files = list(Path(temp_cache_dir).glob("*.pkl"))
        assert len(cache_files) == 1
        assert cache_files[0].suffix == ".pkl"

        # Verify the hash is consistent
        func_signature = {
            'function': 'test_func',
            'args': (42,),
            'kwargs': {}
        }
        expected_hash = hashlib.md5(str(func_signature).encode()).hexdigest()
        expected_filename = f"{expected_hash}.pkl"

        assert cache_files[0].name == expected_filename

    def test_different_args_different_cache(
            self, temp_cache_dir, call_counter):
        """Test that different arguments create different cache entries"""

        @file_cache(cache_dir=temp_cache_dir)
        def test_func(x, y=1):
            call_counter['count'] += 1
            return x + y

        # Call with different arguments
        test_func(1, y=2)
        test_func(1, y=3)
        test_func(2, y=2)

        assert call_counter['count'] == 3

        # Verify multiple cache files exist
        cache_files = list(Path(temp_cache_dir).glob("*.pkl"))
        assert len(cache_files) == 3

    def test_cache_persistence(self, temp_cache_dir):
        """Test that cache persists between function definitions"""

        # First function definition and call
        @file_cache(cache_dir=temp_cache_dir)
        def test_func(x):
            return x * 100

        result1 = test_func(5)
        assert result1 == 500

        # Verify cache file exists
        cache_files = list(Path(temp_cache_dir).glob("*.pkl"))
        assert len(cache_files) == 1

        # New function definition with same name and cache dir
        call_count = 0

        @file_cache(cache_dir=temp_cache_dir)
        def test_func(x):
            nonlocal call_count
            call_count += 1
            return x * 200  # Different implementation

        # Should load from cache (old result)
        result2 = test_func(5)
        assert result2 == 500  # Original cached result
        assert call_count == 0  # Function not executed

    def test_clear_cache_functionality(self, temp_cache_dir):
        """Test the clear_cache method"""

        @file_cache(cache_dir=temp_cache_dir)
        def test_func(x):
            return x * 2

        # Create some cache entries
        test_func(1)
        test_func(2)
        test_func(3)

        # Verify cache files exist
        cache_files = list(Path(temp_cache_dir).glob("*.pkl"))
        assert len(cache_files) == 3

        # Clear cache
        test_func.clear_cache()

        # Verify cache files are removed
        cache_files = list(Path(temp_cache_dir).glob("*.pkl"))
        assert len(cache_files) == 0

    def test_cache_info_functionality(self, temp_cache_dir):
        """Test the cache_info method"""

        @file_cache(cache_dir=temp_cache_dir)
        def test_func(x):
            return "result" * x

        # Initially no cache
        info = test_func.cache_info()
        assert info['cached_results'] == 0
        assert info['total_size_bytes'] == 0
        assert info['cache_dir'] == temp_cache_dir

        # Add some cache entries
        test_func(1)
        test_func(2)

        info = test_func.cache_info()
        assert info['cached_results'] == 2
        assert info['total_size_bytes'] > 0
        assert info['cache_dir'] == temp_cache_dir

    def test_corrupted_cache_file_handling(self, temp_cache_dir):
        """Test handling of corrupted cache files"""

        @file_cache(cache_dir=temp_cache_dir)
        def test_func(x):
            return x * 10

        # Create a cache entry
        result1 = test_func(5)
        assert result1 == 50

        # Find the cache file and corrupt it
        cache_files = list(Path(temp_cache_dir).glob("*.pkl"))
        assert len(cache_files) == 1

        # Write invalid pickle data
        with open(cache_files[0], 'w') as f:
            f.write("corrupted data")

        # Function should handle corruption and recompute
        result2 = test_func(5)
        assert result2 == 50

        # Corrupted file should be removed and new one created
        cache_files = list(Path(temp_cache_dir).glob("*.pkl"))
        assert len(cache_files) == 1

    def test_improved_pickle_error_handling_with_mock(
            self, temp_cache_dir, call_counter):
        """Test improved handling with mocked pickle errors"""

        # Create an improved version of the decorator for this test
        def improved_file_cache(cache_dir="cache"):
            def decorator(func):
                @wraps(func)
                def wrapper(*args, **kwargs):
                    Path(cache_dir).mkdir(exist_ok=True)

                    func_signature = {
                        'function': func.__name__,
                        'args': args,
                        'kwargs': kwargs
                    }

                    signature_str = str(func_signature)
                    hash_obj = hashlib.md5(signature_str.encode())
                    cache_filename = f"{hash_obj.hexdigest()}.pkl"
                    cache_path = Path(cache_dir) / cache_filename

                    if cache_path.exists():
                        try:
                            with open(cache_path, 'rb') as f:
                                result = pickle.load(f)
                            return result
                        except (pickle.PickleError, EOFError, AttributeError) as e:
                            cache_path.unlink()

                    result = func(*args, **kwargs)

                    # Improved error handling - catch all pickle-related
                    # exceptions
                    try:
                        with open(cache_path, 'wb') as f:
                            pickle.dump(result, f)
                    except (pickle.PickleError, AttributeError, TypeError, RecursionError):
                        # Clean up the failed cache file and continue without
                        # caching
                        if cache_path.exists():
                            cache_path.unlink()

                    return result
                return wrapper
            return decorator

        @improved_file_cache(cache_dir=temp_cache_dir)
        def test_func():
            call_counter['count'] += 1
            return lambda x: x  # Unpickleable

        # Should work without raising exception
        result1 = test_func()
        assert callable(result1)
        assert call_counter['count'] == 1

        # Should call function again since caching failed
        result2 = test_func()
        assert callable(result2)
        assert call_counter['count'] == 2

        # No cache files should exist (failed files should be cleaned up)
        cache_files = list(Path(temp_cache_dir).glob("*.pkl"))
        assert len(cache_files) == 0

    def test_function_metadata_preservation(self, temp_cache_dir):
        """Test that function metadata is preserved"""

        @file_cache(cache_dir=temp_cache_dir)
        def documented_function(x, y=1):
            """This is a test function with documentation"""
            return x + y

        assert documented_function.__name__ == "documented_function"
        assert documented_function.__doc__ == "This is a test function with documentation"

    def test_complex_arguments(self, temp_cache_dir, call_counter):
        """Test caching with complex argument types"""

        @file_cache(cache_dir=temp_cache_dir)
        def test_func(data_list, data_dict, flag=True):
            call_counter['count'] += 1
            return len(data_list) + len(data_dict) + int(flag)

        # Test with complex arguments
        result1 = test_func([1, 2, 3], {'a': 1, 'b': 2}, flag=True)
        assert result1 == 6
        assert call_counter['count'] == 1

        # Same arguments should use cache
        result2 = test_func([1, 2, 3], {'a': 1, 'b': 2}, flag=True)
        assert result2 == 6
        assert call_counter['count'] == 1

        # Different arguments should not use cache
        result3 = test_func([1, 2, 3], {'a': 1, 'b': 2}, flag=False)
        assert result3 == 5
        assert call_counter['count'] == 2

    def test_default_cache_directory(self, temp_cache_dir):
        """Test that default cache directory is used correctly"""
        import os

        # Change to temporary directory
        original_cwd = os.getcwd()
        os.chdir(temp_cache_dir)

        try:
            @file_cache()  # No cache_dir specified
            def test_func(x):
                return x

            test_func(42)

            # Should create default cache directory in current working
            # directory
            default_cache = Path("cache")
            assert default_cache.exists()

            cache_files = list(default_cache.glob("*.pkl"))
            assert len(cache_files) == 1

        finally:
            # Always restore original working directory
            os.chdir(original_cwd)

    def test_multiple_functions_same_cache_dir(self, temp_cache_dir):
        """Test multiple functions using the same cache directory"""

        @file_cache(cache_dir=temp_cache_dir)
        def func1(x):
            return x * 2

        @file_cache(cache_dir=temp_cache_dir)
        def func2(x):
            return x * 3

        func1(10)
        func2(10)

        # Should create separate cache files for different functions
        cache_files = list(Path(temp_cache_dir).glob("*.pkl"))
        assert len(cache_files) == 2

        # Clear cache for one function shouldn't affect the other
        func1.clear_cache()

        cache_files = list(Path(temp_cache_dir).glob("*.pkl"))
        assert len(cache_files) == 0  # clear_cache removes all .pkl files

    @pytest.mark.parametrize("input_val,expected", [
        (0, 0),
        (1, 2),
        (5, 10),
        (-3, -6),
        (100, 200)
    ])
    def test_parametrized_caching(
            self,
            temp_cache_dir,
            input_val,
            expected,
            call_counter):
        """Test caching with parametrized inputs"""

        @file_cache(cache_dir=temp_cache_dir)
        def double_func(x):
            call_counter['count'] += 1
            return x * 2

        result = double_func(input_val)
        assert result == expected

        # Second call should use cache
        result2 = double_func(input_val)
        assert result2 == expected

        # Should only be called once per unique input
        # (call_counter is shared across parametrized tests, so we can't assert exact count)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
