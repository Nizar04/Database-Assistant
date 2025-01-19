# Database-Assistant
A Python application that allows users to interact with databases using natural language queries. Available in both GUI and command-line interfaces, with support for CPU and GPU inference.
## Features

### GUI Version (main.py)
- User-friendly graphical interface
- CPU-optimized for widespread compatibility
- ~10 second inference time
- No additional GPU dependencies required

### Command-Line Version (database_assistant_gpu.py)
- Supports both CPU and GPU inference
- ~1 second inference time on GPU
- ~10 second inference time on CPU
- Lightweight command-line interface
- Automatic GPU detection and selection

### Common Features
- Support for multiple database types.
- Natural language to SQL query conversion using PreMai 1B SQL model
- Real-time query execution and results display
- Clean resource management

### Supported Databases
- SQLite (file-based)
- PostgreSQL
- MySQL
- MariaDB

## Requirements

- Python 3.8+
- PyQt6
- torch (CPU version)
- transformers
- pandas
- sqlite3
- psycopg2 (for PostgreSQL)
- mysql-connector-python (for MySQL/MariaDB)
- 
### Basic Installation (CPU only, GUI version)
```bash
pip install -r requirements.txt
```

### GPU Support (Command-line version)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Usage

### GUI Version
```bash
python main.py
```

### Command-Line Version
```bash
python database_assistant.py
```
- If a CUDA-compatible GPU is detected, you'll be prompted to choose between GPU and CPU inference
- Follow the interactive prompts to connect to your database and start querying

## Project Structure

```
database-assistant/
├── main.py                 # GUI application
├── database_assistant.py
├── database_assistant_gpu.py      
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

Developed by El Mouaquit Nizar

---

## Development Notes

### Database Connection Classes

The project implements an abstract `DatabaseConnection` class with specific implementations for:
- SQLite
- PostgreSQL
- MySQL/MariaDB

Each implementation handles:
- Connection management
- Query execution
- Schema extraction
- Resource cleanup

### Query Generation

The application uses the PreMai 1B SQL model for converting natural language to SQL queries. Key points:
- CPU-based inference (~10s response time)
- No GPU dependencies required
- Efficient memory management
- Automatic resource cleanup

### GUI Features

The PyQt6-based GUI provides:
- Database connection setup interface
- Query input and display areas
- Results table with automatic column sizing
- Comprehensive error handling

## Future Improvements

- Add support for more database types
- Implement query history
- Add export functionality for results
- Enhance error handling
- Add unit tests and integration tests
- Implement query optimization suggestions
- Optional GPU support within GUI (separate branch)
```
