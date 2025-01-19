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
        self.connection = psycopg2.connect(**connection_params)

    def execute_query(self, query: str) -> pd.DataFrame:
        return pd.read_sql_query(query, self.connection)

    def get_schema(self) -> str:
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public';
        """)
        tables = cursor.fetchall()

        schema = []
        for table in tables:
            table_name = table[0]
            cursor.execute(f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position;
            """)
            columns = cursor.fetchall()
            column_defs = [f"{col[0]} {col[1]}" for col in columns]
            schema.append(f"CREATE TABLE {table_name} (\n  " +
                          ",\n  ".join(column_defs) + "\n);")

        return "\n\n".join(schema)

    def close(self) -> None:
        if self.connection:
            self.connection.close()


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
    def __init__(self, model_path: str = "premai-io/prem-1B-SQL", device: str = "cpu"):
        """Initialize the Database Assistant with the specified model and device."""
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = device
        self.db_connection = None
        self.current_schema = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the model and move it to the appropriate device."""
        try:
            print("Initializing Database Assistant...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            print("Tokenizer loaded.")
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
            print("Model loaded.")
            self.model = self.model.to(self.device)
            print(f"Model loaded successfully on {self.device}")
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
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
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


def get_connection_params(db_type: str) -> Dict[str, Any]:
    """Get database connection parameters from user input."""
    params = {}

    if db_type == 'sqlite':
        params['database'] = input("Enter the path to your SQLite database: ")
    else:
        params['host'] = input("Enter host (default: localhost): ") or 'localhost'
        params['port'] = input(f"Enter port (default: {5432 if db_type == 'postgresql' else 3306}): ")
        params['database'] = input("Enter database name: ")
        params['user'] = input("Enter username: ")
        params['password'] = input("Enter password: ")

        # Convert port to integer if provided
        if params['port']:
            params['port'] = int(params['port'])
        else:
            params['port'] = 5432 if db_type == 'postgresql' else 3306

    return params


def main():
    print("Welcome to the Database Assistant!")

    # Check for GPU availability
    if torch.cuda.is_available():
        device_choice = input("Do you want to run the model on GPU(1) or CPU(2): ").strip().lower()
        device = "cuda" if device_choice == '1' else "cpu"
    else:
        print("No CUDA-compatible GPU found. Defaulting to CPU.")
        device = "cpu"

    assistant = DatabaseAssistant(device=device)

    # Database type selection
    while True:
        print("\nSupported database types:")
        print("1. SQLite")
        print("2. PostgreSQL")
        print("3. MySQL")
        print("4. MariaDB")

        db_choice = input("Select database type (1-4): ").strip()
        db_types = {
            '1': 'sqlite',
            '2': 'postgresql',
            '3': 'mysql',
            '4': 'mariadb'
        }

        if db_choice in db_types:
            db_type = db_types[db_choice]
            break
        else:
            print("Invalid choice. Try again.")

    connection_params = get_connection_params(db_type)
    assistant.connect_to_database(db_type, connection_params)

    while True:
        # Get natural language query from user
        natural_query = input("Enter your SQL query in natural language or 'exit' to quit: ")

        if natural_query.lower() == "exit":
            break

        try:
            # Generate the SQL query
            query = assistant.generate_query(natural_query)
            print(f"Generated SQL Query: {query}")

            # Execute the query and show results
            results = assistant.execute_query(query)
            print("Query Results:")
            print(results)
        except Exception as e:
            print(f"Error: {str(e)}")

    assistant.clean_up()


if __name__ == "__main__":
    main()
