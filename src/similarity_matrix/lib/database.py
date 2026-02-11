
import mysql.connector
from mysql.connector import Error

from similarity_matrix.lib.logging import logger


# -----------------------------------------------------------------------
# Database

class Database:
    """
    A context manager for MySQL database connection.

    After creating an instance of this class, you can use it as a context manager.
    The connection will be established when entering the context and closed when exiting.

        # Example:
            ```python
            with Database(host, user, password, database) as cursor:
                # Use the cursor to execute queries
                cursor.execute("SELECT * FROM table")
                result = cursor.fetchall()
                # ...
            ```
    """

    def __init__(
            self,
            host: str,
            port: int,
            user: str,
            password: str,
            database: str):
        self.config = {
            'host': host,
            'port': port,
            'user': user,
            'password': password,
            'database': database,
        }
        self.conn = None
        self.cursor = None

    def __enter__(self):
        try:
            try:
                self.conn = mysql.connector.connect(**self.config)
            except mysql.connector.errors.DatabaseError as e:
                # FIXME: better handling of "Unknown collation" error
                # MariaDB does throws an error without additional params
                # 1273 (HY000): Unknown collation: 'utf8mb4_0900_ai_ci'
                self.conn = mysql.connector.connect(
                    **self.config,
                    charset='utf8mb4',
                    collation='utf8mb4_general_ci')
            self.cursor = self.conn.cursor(dictionary=True)
            return self.cursor
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cursor:
            self.cursor.close()
        if self.conn:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
            self.conn.close()

    def test_connection(self) -> Error | None:
        """Test the connection to the MySQL database."""
        try:
            with self as cursor:
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                if result:
                    logger.debug("Connection to MySQL database successful.")
                else:
                    logger.error("Connection to MySQL database failed.")
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            return e
