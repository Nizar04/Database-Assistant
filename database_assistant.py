import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sqlite3
import psycopg2
import mysql.connector
import pandas as pd
import os
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import urllib.parse


class DatabaseConnection(ABC):
    """Abstract base class for database connections."""

    @abstractmethod
    def connect(self, connection_params: Dict[str, Any]) -> None:
        """Establish connection to the database."""
        pass

    @abstractmethod
    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute a query and return results as DataFrame."""
        pass

    @abstractmethod
    def get_schema(self) -> str:
        """Extract and return the database schema."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the database connection."""
        pass


class SQLiteConnection(DatabaseConnection):
    def __init__(self):
        self.connection = None

    def connect(self, connection_params: Dict[str, Any]) -> None:
        if not os.path.exists(connection_params['database']):
            raise FileNotFoundError(f"Database at {connection_params['database']} does not exist.")
        self.connection = sqlite3.connect(connection_params['database'])

    def execute_query(self, query: str) -> pd.DataFrame:
        return pd.read_sql_query(query, self.connection)

    def get_schema(self) -> str:
        cursor = self.connection.cursor()
        tables = cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name NOT LIKE 'sqlite_%';
        """).fetchall()

        schema = []
        for table in tables:
            table_name = table[0]
            columns = cursor.execute(f"PRAGMA table_info('{table_name}')").fetchall()
            column_defs = [f"{col[1]} {col[2]}" for col in columns]
            schema.append(f"CREATE TABLE {table_name} (\n  " +
                          ",\n  ".join(column_defs) + "\n);")

        return "\n\n".join(schema)

    def close(self) -> None:
        if self.connection:
            self.connection.close()


class PostgreSQLConnection(DatabaseConnection):
    def __init__(self):
        self.connection = None

    def connect(self, connection_params: Dict[str, Any]) -> None:
        try:
            self.connection = psycopg2.connect(**connection_params)
        except psycopg2.Error as e:
            raise RuntimeError(f"PostgreSQL connection failed: {str(e)}")

    def get_schema(self) -> str:
        try:
            cursor = self.connection.cursor()

            # First, check if we can access information_schema
            cursor.execute("""
                SELECT has_schema_privilege('information_schema', 'usage');
            """)
            has_access = cursor.fetchone()[0]
            if not has_access:
                raise RuntimeError("No access to information_schema")

            # Get all schemas user has access to
            cursor.execute("""
                SELECT schema_name 
                FROM information_schema.schemata 
                WHERE schema_name NOT IN ('information_schema', 'pg_catalog', 'pg_toast');
            """)
            schemas = [row[0] for row in cursor.fetchall()]

            if not schemas:
                raise RuntimeError("No accessible schemas found")

            all_tables = []
            for schema in schemas:
                # Get tables for each accessible schema
                cursor.execute(f"""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = %s 
                    AND table_type = 'BASE TABLE';
                """, (schema,))
                tables = cursor.fetchall()

                for table in tables:
                    table_name = table[0]
                    qualified_table = f"{schema}.{table_name}"

                    # Get column information
                    cursor.execute(f"""
                        SELECT 
                            column_name,
                            data_type,
                            character_maximum_length,
                            numeric_precision,
                            is_nullable,
                            column_default
                        FROM information_schema.columns 
                        WHERE table_schema = %s 
                        AND table_name = %s 
                        ORDER BY ordinal_position;
                    """, (schema, table_name))

                    columns = cursor.fetchall()
                    column_defs = []

                    for col in columns:
                        col_name = col[0]
                        data_type = col[1]
                        max_length = col[2]
                        precision = col[3]
                        nullable = col[4]
                        default = col[5]

                        # Build column definition
                        col_def = f"{col_name} {data_type}"
                        if max_length:
                            col_def += f"({max_length})"
                        elif precision:
                            col_def += f"({precision})"
                        if default:
                            col_def += f" DEFAULT {default}"
                        if nullable == 'NO':
                            col_def += " NOT NULL"

                        column_defs.append(col_def)

                    table_sql = f"CREATE TABLE {qualified_table} (\n  " + \
                                ",\n  ".join(column_defs) + "\n);"
                    all_tables.append(table_sql)

            return "\n\n".join(all_tables)

        except psycopg2.Error as e:
            raise RuntimeError(f"Failed to get schema: {str(e)}")
        finally:
            if cursor:
                cursor.close()

    def execute_query(self, query: str) -> pd.DataFrame:
        try:
            return pd.read_sql_query(query, self.connection)
        except psycopg2.Error as e:
            raise RuntimeError(f"Query execution failed: {str(e)}")

    def close(self) -> None:
        if self.connection:
            try:
                self.connection.close()
            except psycopg2.Error:
                pass


class MySQLConnection(DatabaseConnection):
    def __init__(self):
        self.connection = None

    def connect(self, connection_params: Dict[str, Any]) -> None:
        self.connection = mysql.connector.connect(**connection_params)

    def execute_query(self, query: str) -> pd.DataFrame:
        return pd.read_sql_query(query, self.connection)

    def get_schema(self) -> str:
        cursor = self.connection.cursor()
        cursor.execute("SHOW TABLES;")
        tables = cursor.fetchall()

        schema = []
        for table in tables:
            table_name = table[0]
            cursor.execute(f"DESCRIBE {table_name};")
            columns = cursor.fetchall()
            column_defs = [f"{col[0]} {col[1]}" for col in columns]
            schema.append(f"CREATE TABLE {table_name} (\n  " +
                          ",\n  ".join(column_defs) + "\n);")

        return "\n\n".join(schema)

    def close(self) -> None:
        if self.connection:
            self.connection.close()


class DatabaseAssistant:
    def __init__(self, model_path: str = "premai-io/prem-1B-SQL"):  # Changed default model
        """Initialize the Database Assistant with the specified model."""
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.db_connection = None
        self.current_schema = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the model."""
        try:
            print("Initializing Database Assistant...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            print("Tokenizer loaded.")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                rope_scaling={"type": "linear", "factor": 4.0}
            )
            print("Model loaded.")
            # Force CPU usage
            self.model = self.model.to('cpu')
            print("Model loaded successfully on CPU")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {str(e)}")

    def connect_to_database(self, db_type: str, connection_params: Dict[str, Any]) -> None:
        """Connect to a database of the specified type using the provided parameters."""
        db_connections = {
            'sqlite': SQLiteConnection,
            'postgresql': PostgreSQLConnection,
            'mysql': MySQLConnection,
            'mariadb': MySQLConnection  # MariaDB uses the same connector as MySQL
        }

        if db_type not in db_connections:
            raise ValueError(f"Unsupported database type: {db_type}")

        try:
            connection_class = db_connections[db_type]
            self.db_connection = connection_class()
            self.db_connection.connect(connection_params)
            self.current_schema = self.db_connection.get_schema()
            print(f"Successfully connected to {db_type} database")
        except Exception as e:
            raise RuntimeError(f"Failed to connect to database: {str(e)}")

    def generate_query(self, natural_query: str) -> str:
        """Generate SQL query from natural language."""
        if not self.current_schema:
            raise RuntimeError("No database schema loaded")

        prompt = self._build_prompt(natural_query)

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")

            # Get input length
            input_length = inputs.input_ids.shape[1]

            # Calculate safe max_new_tokens (leaving room for the generated SQL)
            max_new_tokens = 256  # Reasonable length for SQL query

            # Calculate safe total length
            max_total_length = input_length + max_new_tokens

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_total_length,
                    max_new_tokens=max_new_tokens,
                    num_beams=5,
                    early_stopping=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            query = generated_text.split("### SQL Query:")[-1].strip()
            return query
        except Exception as e:
            raise RuntimeError(f"Failed to generate query: {str(e)}")

    def _build_prompt(self, natural_query: str) -> str:
        """Build the prompt for the model."""
        return f"""### SQL Database Schema:
{self.current_schema}

### User Question:
{natural_query}

### SQL Query:"""

    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute the generated SQL query and return results as a DataFrame."""
        if not self.db_connection:
            raise RuntimeError("No database connection established")

        try:
            return self.db_connection.execute_query(query)
        except Exception as e:
            raise RuntimeError(f"Failed to execute query: {str(e)}")

    def clean_up(self) -> None:
        """Clean up memory and close the database connection."""
        if self.db_connection:
            self.db_connection.close()
            print("Database connection closed.")
        if self.model:
            del self.model
            print("Model unloaded.")
        if self.tokenizer:
            del self.tokenizer
            print("Tokenizer unloaded.")