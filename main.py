"""
Launcher: creates the QApplication, applies ui/styles.qss and shows the main window.
"""
import sys
from pathlib import Path

from PyQt6.QtWidgets import QApplication

from ui.main_window import MainWindow

QSS_PATH = Path(__file__).resolve().parent / "ui" / "styles.qss"


def main():
    app = QApplication(sys.argv)

    if not QSS_PATH.is_file():
        raise FileNotFoundError(f"Stylesheet not found: {QSS_PATH}")
    app.setStyleSheet(QSS_PATH.read_text(encoding="utf-8"))

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
