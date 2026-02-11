"""
mock_pipeline.py
----------------
A fully self-contained example pipeline that uses plain dictionaries instead
of a real database. Use it as a starting point or to test the framework
without setting up any infrastructure.

Place this file (or a copy) inside your --pipeline-dir directory and run:

    similarity-matrix --pipeline mock compute-similarity
"""

from similarity_matrix.lib.pipeline import Pipeline
from similarity_matrix.lib.database import Database


# ---------------------------------------------------------------------------
# Fake data — replace these with real database queries in a real pipeline
# ---------------------------------------------------------------------------

_ROWS: dict[int, str] = {
    1: "Introduction to machine learning and neural networks",
    2: "Advanced algorithms and data structures",
    3: "Database design and SQL fundamentals",
    4: "Natural language processing with transformers",
    5: "Operating systems and memory management",
}

_COLUMNS: dict[int, str] = {
    101: "Deep learning and backpropagation techniques",
    102: "Graph algorithms and dynamic programming",
    103: "Relational databases and query optimisation",
    104: "Text classification using large language models",
    105: "Linux kernel internals and process scheduling",
}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class MockPipeline(Pipeline):
    """
    A demo pipeline backed by in-memory dictionaries.

    The row entities are 'courses' and the column entities are 'topics'.
    The similarity matrix will express how closely each course relates to
    each topic based on their text descriptions.
    """

    def __init__(self, db: Database, path: str = "./matrices"):
        # `db` is injected by the framework and kept on self.db, but we
        # don't use it here — all data comes from the dicts above.
        super().__init__(name="mock", db=db, path=path)

    # -----------------------------------------------------------------------
    # IDs
    # -----------------------------------------------------------------------

    def get_row_ids(self) -> list[int]:
        """Return the IDs of all courses (rows)."""
        return list(_ROWS.keys())

    def get_column_ids(self) -> list[int]:
        """Return the IDs of all topics (columns)."""
        return list(_COLUMNS.keys())

    # -----------------------------------------------------------------------
    # Text values used to compute similarity
    # -----------------------------------------------------------------------

    def get_row_values(self) -> list[str]:
        """Return the text description for each course, in the same order as get_row_ids()."""
        return [_ROWS[row_id] for row_id in self.get_row_ids()]

    def get_column_values(self) -> list[str]:
        """Return the text description for each topic, in the same order as get_column_ids()."""
        return [_COLUMNS[col_id] for col_id in self.get_column_ids()]

    # -----------------------------------------------------------------------
    # Database update stubs
    # -----------------------------------------------------------------------

    def update_db_row_table(self):
        """
        In a real pipeline this would upsert rows into the 'courses' table.
        Here we just print what would be written.
        """
        print("[mock] update_db_row_table — rows that would be upserted:")
        for row_id, text in _ROWS.items():
            print(
                f"  INSERT INTO courses (id, description) VALUES ("
                f"{row_id!r}, {text!r})")

    def update_db_column_table(self):
        """
        In a real pipeline this would upsert rows into the 'topics' table.
        Here we just print what would be written.
        """
        print("[mock] update_db_column_table — rows that would be upserted:")
        for col_id, text in _COLUMNS.items():
            print(
                f"  INSERT INTO topics (id, description) VALUES ("
                f"{col_id!r}, {text!r})")

    def update_db_matrix_table(self):
        """
        In a real pipeline this would write similarity scores to a 'course_topic_similarity'
        table. Here we load the matrix and print the scores that would be written.
        """
        sm = self.get_matrix()
        print("[mock] update_db_matrix_table — scores that would be upserted:")
        for i, row_id in enumerate(self.get_row_ids()):
            for j, col_id in enumerate(self.get_column_ids()):
                score = sm.data[i, j]
                print(
                    f"  INSERT INTO course_topic_similarity "
                    f"(course_id, topic_id, score) VALUES ({row_id}, {col_id}, {score:.4f})"
                )

    # -----------------------------------------------------------------------
    # Optional: postprocess the matrix before it is saved
    # -----------------------------------------------------------------------

    def postprocess_matrix(self):
        """
        Example postprocessing: clamp all scores below 0.1 to 0.0 to remove
        noise from very dissimilar pairs.
        """
        self._sm.data[self._sm.data < 0.1] = 0.0
        return self._sm
