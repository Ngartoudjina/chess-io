"""Lancement de l'application graphique."""

from __future__ import annotations

import os
import sys

from PySide6.QtCore import QRect, QSettings, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPixmap
from PySide6.QtWidgets import QApplication, QMessageBox, QSplashScreen, QWidget

from chessio import __version__
from chessio.ui.style import PALETTE, app_icon, logo_pixmap, stylesheet
from chessio.ui.widgets import fit_to_screen
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
    # Le thème GTK de Qt lit un réglage GNOME absent d'Ubuntu 26.04 et fait planter
    # l'application ; Chessio a son propre style, le thème générique suffit.
    os.environ.setdefault("QT_QPA_PLATFORMTHEME", "generic")
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Chessio")
    app.setOrganizationName("chessio")
    app.setApplicationDisplayName("Chessio")
    app.setDesktopFileName("chessio")  # relie les fenêtres au lanceur (icône du dock)
    app.setStyle("Fusion")  # base identique partout, que la feuille de style habille
    app.setStyleSheet(stylesheet())
    app.setWindowIcon(app_icon())
    return app


def splash_screen() -> QSplashScreen:
    """Écran d'accueil pendant le chargement (moteur, réseaux de neurones)."""
    pixmap = QPixmap(460, 300)
    pixmap.fill(QColor(PALETTE["bg"]))
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.drawPixmap(170, 40, logo_pixmap(120))
    font = QFont()
    font.setPixelSize(30)
    font.setBold(True)
    painter.setFont(font)
    painter.setPen(QColor(PALETTE["text"]))
    painter.drawText(QRect(0, 175, 460, 40), Qt.AlignmentFlag.AlignCenter, "Chessio")
    font.setPixelSize(14)
    font.setBold(False)
    painter.setFont(font)
    painter.setPen(QColor(PALETTE["muted"]))
    painter.drawText(
        QRect(0, 215, 460, 24), Qt.AlignmentFlag.AlignCenter, "Le meilleur coup, en un geste"
    )
    painter.end()
    splash = QSplashScreen(pixmap)
    splash.showMessage(
        f"Version {__version__} · chargement…",
        Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter,
        QColor(PALETTE["faint"]),
    )
    return splash


def main() -> int:
    app = create_app()
    if not confirm_warning():  # première utilisation seulement
        return 0
    splash = splash_screen()
    splash.show()
    app.processEvents()
    window = MainWindow()
    fit_to_screen(window, 1320, 840)
    window.show()
    splash.finish(window)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
