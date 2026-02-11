
import re
from mysql.connector.cursor import MySQLCursor
from typing import Iterator, Iterable, TypeVar, List

from similarity_matrix.lib.matrix import SimilarityMatrix


# -----------------------------------------------------------------------
# Utils

T = TypeVar('T')


def _batch_iterator(iterator: Iterable[T],
                    batch_size: int) -> Iterator[List[T]]:
    """
    Yields batches of items from an iterator.

    Args:
        iterator: An iterator to batch.
        batch_size: The number of items in each batch.

    Returns:
        An iterator of batches of items.
    """
    batch: List[T] = []
    for item in iterator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def update_db_batches(
        cursor: MySQLCursor,
        sql: str,
        matrix: SimilarityMatrix,
        batch_size: int = 1000):
    """
    Update the skill similarity table in batches.

    Args:
        cursor: A MySQL cursor.
        sql: The SQL query to execute.
        matrix: The similarity matrix to update.
        batch_size: The number of rows to insert at a time.

    Raises:
        Exception: If the SQL query fails.
    """
    # Check if the matrix is empty
    if matrix.num_rows == 0 or matrix.num_columns == 0:
        return

    for batch in _batch_iterator(
            ((row_id, column_id, float(similarity))
             for row_id, column_id, similarity in matrix), batch_size):
        cursor.execute("START TRANSACTION")
        try:
            cursor.executemany(sql, batch)
            cursor.execute("COMMIT")
        except Exception as e:
            cursor.execute("ROLLBACK")
            raise e


def camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case"""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def total_length_lists(lists: list[list]) -> int:
    """
    Calculate the total length of all lists in a list of lists.

    Args:
        lists: List of lists to calculate total length for

    Returns:
        Total length of all lists
    """
    return sum(len(lst) for lst in lists) if lists else 0
