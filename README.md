# similarity-matrix

A command-line tool to compute similarity between two sets of texts and organizing the results in a matrix of cosine-similarities.

## Installation

```bash
pip install similarity-matrix
```

## Configuration

Database credentials can be provided via CLI flags or a `.env` file:

```env
DB_USERNAME=your_username
DB_PASSWORD=your_password
```

## Usage

```
similarity-matrix [options] <action> [action options]
```

### Global Options

| Option | Default | Description |
|---|---|---|
| `--db-host` | `localhost` | Database host |
| `--db-port` | `3306` | Database port |
| `--db-name` | `universitaly` | Database name |
| `--db-user` | — | Database user (or via `DB_USERNAME` in `.env`) |
| `--db-password` | — | Database password (or via `DB_PASSWORD` in `.env`) |
| `--pipeline-dir` | `./pipelines` | Path to directory containing pipeline files |
| `--pipeline` | — | Name of the pipeline to use (`.py` filename without extension) |
| `--output-dir` | `./matrices` | Directory where similarity matrices will be saved |
| `--list-pipelines` | — | List all available pipelines and exit |
| `--debug` | — | Enable debug logging |

### Actions

#### `update-db-row`
Updates the row table in the database for the selected pipeline.

```bash
similarity-matrix --pipeline my_pipeline update-db-row
```

#### `update-db-column`
Updates the column table in the database for the selected pipeline.

```bash
similarity-matrix --pipeline my_pipeline update-db-column
```

#### `update-db-matrix`
Updates the matrix table in the database for the selected pipeline.

```bash
similarity-matrix --pipeline my_pipeline update-db-matrix
```

#### `compute-similarity`
Computes the similarity matrix for the selected pipeline and saves it to the output directory.

```bash
similarity-matrix --pipeline my_pipeline --output-dir ./results compute-similarity
```

### Examples

List all available pipelines:
```bash
similarity-matrix --list-pipelines
```

Run a full update using a custom database and pipeline directory:
```bash
similarity-matrix \
  --db-host db.example.com \
  --db-name mydb \
  --db-user admin \
  --pipeline-dir ./my_pipelines \
  --pipeline course_similarity \
  update-db-matrix
```

Compute similarity and save results to a specific directory:
```bash
similarity-matrix \
  --pipeline course_similarity \
  --output-dir ./output/matrices \
  compute-similarity
```

## Build a Pipeline

A pipeline is a Python file placed in your `--pipeline-dir` directory. It must define a class whose name ends with `Pipeline` and that subclasses `Pipeline` from `similarity_matrix.lib.pipeline`. The filename (without `.py`) is used as the pipeline's identifier when passing `--pipeline` on the command line.

### Concepts

The framework models a similarity matrix as a grid where **rows** and **columns** represent two sets of entities (they can be the same set). Each entity has an **ID** and an associated **text value**. The matrix cell at `[i, j]` holds the similarity score between row entity `i` and column entity `j`.

Beyond computing the matrix, a pipeline is also responsible for keeping three database tables in sync:

- **Row table** — holds the data for the entities on the rows.
- **Column table** — holds the data for the entities on the columns.
- **Matrix table** — holds the computed similarity scores.

### Abstract methods to implement

Every pipeline must implement these six methods:

| Method | What it should return |
|---|---|
| `get_row_ids()` | A list of IDs for the row entities |
| `get_column_ids()` | A list of IDs for the column entities |
| `get_row_values()` | A list of strings (one per row ID) used to compute similarity, or a `(values, pointers)` tuple for windowed comparison |
| `get_column_values()` | Same as above, for column entities |
| `update_db_row_table()` | Logic to upsert the row table in the database |
| `update_db_column_table()` | Logic to upsert the column table in the database |
| `update_db_matrix_table()` | Logic to upsert the matrix table in the database |

### Optional: windowed values

When dealing with long texts, `get_row_values()` and `get_column_values()` can return a tuple `(values, pointers)` instead of a plain list. `values` is the flat list of text windows (chunks of the original texts), and `pointers` is a list of index ranges that map each window back to its source entity. The framework uses these pointers to aggregate per-window scores back into the expected matrix shape.

### Optional: `postprocess_matrix`

After the matrix is computed but before it is saved to disk, `postprocess_matrix(self) -> SimilarityMatrix` is called. Override it to normalise scores, zero out a diagonal, apply a threshold, etc. The default implementation is a no-op that returns `self._sm` unchanged.

### Optional: chunked computation

Pass `chunk_size=N` to the base `__init__` to use `ChunkedSimilarityMatrix`, which processes the matrix in `N`-sized blocks. This is useful when the full matrix would not fit in memory.

### Discovery

At startup the tool scans every `.py` file in `--pipeline-dir` (excluding `__init__.py`), imports each one, and registers any class whose name ends with `Pipeline`. The class is keyed by its name converted to `snake_case` with the `_pipeline` suffix stripped — so `CourseSimilarityPipeline` in `course_similarity.py` is available as `--pipeline course_similarity`.

### Minimal example

```python
from similarity_matrix.lib.pipeline import Pipeline
from similarity_matrix.lib.database import Database


class MyPipeline(Pipeline):

    def __init__(self, db: Database, path: str = './matrices'):
        super().__init__(name='my', db=db, path=path)

    # --- IDs ---

    def get_row_ids(self) -> list:
        return self.db.query("SELECT id FROM rows_table")

    def get_column_ids(self) -> list:
        return self.db.query("SELECT id FROM columns_table")

    # --- Values ---

    def get_row_values(self) -> list[str]:
        return self.db.query("SELECT text FROM rows_table ORDER BY id")

    def get_column_values(self) -> list[str]:
        return self.db.query("SELECT text FROM columns_table ORDER BY id")

    # --- DB updates ---

    def update_db_row_table(self):
        self.db.execute("INSERT INTO rows_table ...")

    def update_db_column_table(self):
        self.db.execute("INSERT INTO columns_table ...")

    def update_db_matrix_table(self):
        matrix = self.get_matrix()
        self.db.execute("INSERT INTO matrix_table ...", matrix.data)
```

See [`mock_pipeline.py`](pipelines/mock_pipeline.py) for a fully self-contained example that uses plain dictionaries instead of a real database.

## License

See [LICENSE](LICENSE) for details.
