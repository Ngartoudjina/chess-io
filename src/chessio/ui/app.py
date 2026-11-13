"""Lancement de l'application graphique."""

from __future__ import annotations

import sys

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QMessageBox, QWidget

from chessio.ui.style import app_icon, stylesheet
from chessio.ui.window import WARNING, MainWindow


def confirm_warning(parent: QWidget | None = None) -> bool:
    """Avertissement d'usage, affiché jusqu'à acceptation (cahier des charges, section 2)."""
    settings = QSettings("chessio", "chessio")
    if settings.value("warning_accepted", False, type=bool):
        return True
    box = QMessageBox(parent)
    box.setWindowTitle("Cadre d'utilisation")
    box.setIcon(QMessageBox.Icon.Information)
    box.setText(WARNING)
    accept = box.addButton("J'ai compris, continuer", QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Quitter", QMessageBox.ButtonRole.RejectRole)
    box.exec()
    if box.clickedButton() is not accept:
        return False
    settings.setValue("warning_accepted", True)
    return True


def create_app(argv: list[str] | None = None) -> QApplication:
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Chessio")
    app.setOrganizationName("chessio")
    app.setStyle("Fusion")  # base identique partout, que la feuille de style habille
    app.setStyleSheet(stylesheet())
    app.setWindowIcon(app_icon())
    return app


def main() -> int:
    app = create_app()
    if not confirm_warning():
        return 0
    window = MainWindow()
    window.resize(1320, 840)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
