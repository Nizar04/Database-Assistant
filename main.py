import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QComboBox, QLineEdit,
                             QPushButton, QTextEdit, QTableWidget, QTableWidgetItem,
                             QMessageBox, QFileDialog, QStackedWidget)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QMovie
import pandas as pd
from typing import Dict, Any
from database_assistant import DatabaseAssistant


class QueryGenerationWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, assistant, query):
        super().__init__()
        self.assistant = assistant
        self.query = query

    def run(self):
        try:
            sql_query = self.assistant.generate_query(self.query)
            self.finished.emit(sql_query)
        except Exception as e:
            self.error.emit(str(e))


class LoadingSpinner(QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(50, 50)
        self.movie = QMovie("loading.gif")  # Make sure to include a loading.gif in your project
        self.movie.setScaledSize(self.size())
        self.setMovie(self.movie)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hide()

    def start_animation(self):
        self.show()
        self.movie.start()

    def stop_animation(self):
        self.movie.stop()
        self.hide()


class WelcomeScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        title_label = QLabel('Database Assistant')
        title_label.setStyleSheet("font-size: 38px; font-weight: bold;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        developer_label = QLabel('Developed by El Mouaquit Nizar')
        developer_label.setStyleSheet("font-size: 16px;")
        developer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addStretch()
        layout.addWidget(title_label)
        layout.addWidget(developer_label)
        layout.addStretch()

        self.setLayout(layout)


class DatabaseAssistantGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.assistant = None
        self.connection_fields = {}
        self.worker = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Database Assistant')
        self.setFixedSize(800, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        self.stacked_widget = QStackedWidget()
        layout.addWidget(self.stacked_widget)

        self.setup_welcome_page()
        self.setup_setup_page()
        self.setup_main_page()

        self.stacked_widget.setCurrentIndex(0)

    def setup_welcome_page(self):
        welcome_page = WelcomeScreen()
        self.stacked_widget.addWidget(welcome_page)
        QTimer.singleShot(3000, lambda: self.stacked_widget.setCurrentIndex(1))

    def setup_setup_page(self):
        setup_page = QWidget()
        layout = QVBoxLayout(setup_page)
        layout.setSpacing(20)

        # Database type selection
        db_layout = QHBoxLayout()
        db_label = QLabel('Database Type:')
        db_label.setFixedWidth(100)
        self.db_combo = QComboBox()
        self.db_combo.addItems(['SQLite', 'PostgreSQL', 'MySQL', 'MariaDB'])
        db_layout.addWidget(db_label)
        db_layout.addWidget(self.db_combo)
        db_layout.addStretch()
        layout.addLayout(db_layout)

        # Connection parameters container
        self.conn_widget = QWidget()
        self.conn_layout = QVBoxLayout(self.conn_widget)
        self.conn_layout.setSpacing(15)
        layout.addWidget(self.conn_widget)

        # Create all possible connection fields
        self.create_all_connection_fields()

        # Connect the database type change event
        self.db_combo.currentTextChanged.connect(self.update_connection_fields)

        layout.addStretch()

        # Connect button
        self.connect_btn = QPushButton('Connect to Database')
        self.connect_btn.setFixedWidth(200)
        connect_btn_layout = QHBoxLayout()
        connect_btn_layout.addWidget(self.connect_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addLayout(connect_btn_layout)
        self.connect_btn.clicked.connect(self.connect_to_database)

        # Initialize visibility of connection fields
        self.update_connection_fields()

        self.stacked_widget.addWidget(setup_page)

    def create_all_connection_fields(self):
        # Create SQLite fields container
        sqlite_container = QWidget()
        sqlite_layout = QVBoxLayout(sqlite_container)
        sqlite_layout.setContentsMargins(20, 10, 20, 10)

        file_layout = QHBoxLayout()
        file_layout.setSpacing(10)
        self.connection_fields['sqlite'] = {
            'container': sqlite_container,
            'layout': file_layout,
            'path': QLineEdit(),
            'browse': QPushButton('Browse')
        }
        self.connection_fields['sqlite']['browse'].clicked.connect(self.browse_db_file)
        self.connection_fields['sqlite']['browse'].setFixedWidth(100)

        file_label = QLabel('Database File:')
        file_label.setFixedWidth(100)
        file_layout.addWidget(file_label)
        file_layout.addWidget(self.connection_fields['sqlite']['path'])
        file_layout.addWidget(self.connection_fields['sqlite']['browse'])
        sqlite_layout.addLayout(file_layout)
        self.conn_layout.addWidget(sqlite_container)

        # Create other database fields container
        other_container = QWidget()
        other_layout = QVBoxLayout(other_container)
        other_layout.setContentsMargins(20, 10, 20, 10)
        other_layout.setSpacing(10)

        self.connection_fields['other'] = {
            'container': other_container,
            'layout': other_layout,
            'host': QLineEdit('localhost'),
            'port': QLineEdit(),
            'database': QLineEdit(),
            'username': QLineEdit(),
            'password': QLineEdit()
        }

        self.connection_fields['other']['password'].setEchoMode(QLineEdit.EchoMode.Password)

        field_labels = [
            ('Host:', 'host'),
            ('Port:', 'port'),
            ('Database:', 'database'),
            ('Username:', 'username'),
            ('Password:', 'password')
        ]

        for label_text, field_key in field_labels:
            field_layout = QHBoxLayout()
            field_layout.setSpacing(10)

            label = QLabel(label_text)
            label.setFixedWidth(100)

            field_layout.addWidget(label)
            field_layout.addWidget(self.connection_fields['other'][field_key])

            other_layout.addLayout(field_layout)

        self.conn_layout.addWidget(other_container)

    def setup_main_page(self):
        main_page = QWidget()
        layout = QVBoxLayout(main_page)
        layout.setSpacing(15)

        # Query input
        query_label = QLabel('Enter your query in natural language:')
        layout.addWidget(query_label)

        self.query_input = QTextEdit()
        self.query_input.setMaximumHeight(100)
        layout.addWidget(self.query_input)

        # Generated SQL display
        sql_label = QLabel('Generated SQL:')
        layout.addWidget(sql_label)

        self.sql_display = QTextEdit()
        self.sql_display.setMaximumHeight(100)
        self.sql_display.setReadOnly(True)
        layout.addWidget(self.sql_display)

        # Loading spinner
        spinner_layout = QHBoxLayout()
        self.loading_spinner = LoadingSpinner()
        spinner_layout.addWidget(self.loading_spinner)
        spinner_layout.addStretch()
        layout.addLayout(spinner_layout)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)

        self.generate_btn = QPushButton('Generate SQL')
        self.generate_btn.setFixedWidth(150)
        self.generate_btn.clicked.connect(self.generate_sql)
        button_layout.addWidget(self.generate_btn)

        self.execute_btn = QPushButton('Execute Query')
        self.execute_btn.setFixedWidth(150)
        self.execute_btn.clicked.connect(self.execute_sql)
        button_layout.addWidget(self.execute_btn)

        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Results table
        self.results_table = QTableWidget()
        layout.addWidget(self.results_table)

        self.stacked_widget.addWidget(main_page)

    def update_connection_fields(self):
        """Update the visibility of connection fields based on selected database type."""
        db_type = self.db_combo.currentText().lower()

        # Hide both containers first
        self.connection_fields['sqlite']['container'].hide()
        self.connection_fields['other']['container'].hide()

        # Show only the relevant container
        if db_type == 'sqlite':
            self.connection_fields['sqlite']['container'].show()
        else:
            # Set appropriate default port
            self.connection_fields['other']['port'].setText('5432' if db_type == 'postgresql' else '3306')
            self.connection_fields['other']['container'].show()

    def get_connection_params(self) -> Dict[str, Any]:
        """Get connection parameters based on the selected database type."""
        db_type = self.db_combo.currentText().lower()
        params = {}

        if db_type == 'sqlite':
            params['database'] = self.connection_fields['sqlite']['path'].text()
        else:
            params.update({
                'host': self.connection_fields['other']['host'].text(),
                'port': int(self.connection_fields['other']['port'].text()),
                'database': self.connection_fields['other']['database'].text(),
                'user': self.connection_fields['other']['username'].text(),
                'password': self.connection_fields['other']['password'].text()
            })

        return params

    def browse_db_file(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Select SQLite Database', '', 'SQLite Database (*.db *.sqlite);;All Files (*)'
        )
        if filename:
            self.connection_fields['sqlite']['path'].setText(filename)

    def connect_to_database(self):
        try:
            self.assistant = DatabaseAssistant()
            db_type = self.db_combo.currentText().lower()
            connection_params = self.get_connection_params()

            self.assistant.connect_to_database(db_type, connection_params)
            QMessageBox.information(self, 'Success', 'Successfully connected to database!')

            # Switch to main page
            self.stacked_widget.setCurrentIndex(2)

        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to connect to database: {str(e)}')

    def generate_sql(self):
        if not self.assistant:
            QMessageBox.warning(self, 'Warning', 'Please connect to a database first.')
            return

        natural_query = self.query_input.toPlainText()
        if not natural_query:
            QMessageBox.warning(self, 'Warning', 'Please enter a query.')
            return

        # Disable buttons and show loading animation
        self.generate_btn.setEnabled(False)
        self.execute_btn.setEnabled(False)
        self.loading_spinner.start_animation()

        # Create and start worker thread
        self.worker = QueryGenerationWorker(self.assistant, natural_query)
        self.worker.finished.connect(self.on_query_generated)
        self.worker.error.connect(self.on_query_error)
        self.worker.start()

    def on_query_generated(self, sql_query):
        self.sql_display.setText(sql_query)
        self.loading_spinner.stop_animation()
        self.generate_btn.setEnabled(True)
        self.execute_btn.setEnabled(True)

    def on_query_error(self, error_message):
        self.loading_spinner.stop_animation()
        self.generate_btn.setEnabled(True)
        self.execute_btn.setEnabled(True)
        QMessageBox.critical(self, 'Error', f'Failed to generate SQL: {error_message}')

    def execute_sql(self):
        if not self.assistant:
            QMessageBox.warning(self, 'Warning', 'Please connect to a database first.')
            return

        try:
            sql_query = self.sql_display.toPlainText()
            if not sql_query:
                QMessageBox.warning(self, 'Warning', 'Please generate SQL query first.')
                return

            results = self.assistant.execute_query(sql_query)
            self.display_results(results)

        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to execute query: {str(e)}')

    def display_results(self, df: pd.DataFrame):
        # Clear existing items
        self.results_table.clear()

        # Set up table
        self.results_table.setRowCount(len(df))
        self.results_table.setColumnCount(len(df.columns))
        self.results_table.setHorizontalHeaderLabels(df.columns)

        # Fill data
        for i, row in df.iterrows():
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                self.results_table.setItem(i, j, item)

        # Adjust column widths
        self.results_table.resizeColumnsToContents()

    def closeEvent(self, event):
        if self.assistant:
            self.assistant.clean_up()
        event.accept()


def main():
    app = QApplication(sys.argv)
    gui = DatabaseAssistantGUI()
    gui.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()