"""Interface des mises à jour et du partage (menu ⋮)."""

from __future__ import annotations

import os
import sys
import threading

from PySide6.QtCore import QObject, QSettings, Qt, Signal
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QMessageBox, QProgressDialog, QWidget

from chessio import __version__
from chessio.update import (
    RELEASES_PAGE,
    Release,
    UpdateError,
    appimage_path,
    install_update,
    is_newer,
    latest_release,
)

SETTING = "update_check_on_start"


class _Bridge(QObject):
    """Relaie vers le fil de l'interface les résultats obtenus en arrière-plan."""

    checked = Signal(object, bool)  # Release, vérification demandée par l'utilisateur
    failed = Signal(str, bool)
    progress = Signal(int, int)
    installed = Signal(str)


class Updater(QObject):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.parent_widget = parent
        self.bridge = _Bridge()
        self.bridge.checked.connect(self._on_checked)
        self.bridge.failed.connect(self._on_failed)
        self.bridge.installed.connect(self._on_installed)
        self._dialog: QProgressDialog | None = None
        self._cancel = False

    # --- réglage --------------------------------------------------------------------

    @staticmethod
    def check_on_start() -> bool:
        return QSettings("chessio", "chessio").value(SETTING, False, type=bool)

    @staticmethod
    def set_check_on_start(enabled: bool) -> None:
        QSettings("chessio", "chessio").setValue(SETTING, enabled)

    # --- vérification -------------------------------------------------------------

    def check(self, interactive: bool = True) -> None:
        def run() -> None:
            try:
                self.bridge.checked.emit(latest_release(), interactive)
            except UpdateError as exc:
                self.bridge.failed.emit(str(exc), interactive)

        threading.Thread(target=run, name="mise-a-jour", daemon=True).start()

    def _on_failed(self, message: str, interactive: bool) -> None:
        if self._dialog is not None:
            self._dialog.close()
            self._dialog = None
        if interactive:
            QMessageBox.warning(self.parent_widget, "Mise à jour", message)

    def _on_checked(self, release: Release, interactive: bool) -> None:
        if not is_newer(release.version):
            if interactive:
                QMessageBox.information(
                    self.parent_widget,
                    "Mise à jour",
                    f"Chessio est à jour (version {__version__}).",
                )
            return
        notes = release.notes.strip()
        notes = (notes[:600] + "…") if len(notes) > 600 else notes
        target = appimage_path()
        if target is None or release.asset_url is None:
            QMessageBox.information(
                self.parent_widget,
                "Nouvelle version",
                f"La version {release.version} est disponible (vous avez la {__version__}).\n\n"
                f"{notes}\n\nTéléchargez-la sur : {release.page}"
                + ("\n(version de développement : « git pull » suffit)" if target is None else ""),
            )
            return
        answer = QMessageBox.question(
            self.parent_widget,
            "Nouvelle version",
            f"La version {release.version} est disponible (vous avez la {__version__}).\n\n"
            f"{notes}\n\nLa télécharger et l'installer maintenant ?",
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._install(release, target)

    # --- installation --------------------------------------------------------------

    def _install(self, release: Release, target) -> None:
        self._cancel = False
        dialog = QProgressDialog(
            f"Téléchargement de Chessio {release.version}…", "Annuler", 0, 1000, self.parent_widget
        )
        dialog.setWindowTitle("Mise à jour")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.canceled.connect(self._request_cancel)
        self.bridge.progress.connect(self._on_progress)
        dialog.show()
        self._dialog = dialog

        def run() -> None:
            try:
                path = install_update(
                    release,
                    target,
                    progress=lambda done, total: self.bridge.progress.emit(done, total),
                    should_stop=lambda: self._cancel,
                )
                self.bridge.installed.emit(str(path))
            except UpdateError as exc:
                self.bridge.failed.emit(str(exc), True)

        threading.Thread(target=run, name="telechargement", daemon=True).start()

    def _request_cancel(self) -> None:
        self._cancel = True

    def _on_progress(self, done: int, total: int) -> None:
        if self._dialog is not None and total:
            self._dialog.setValue(int(1000 * done / total))
            self._dialog.setLabelText(
                f"Téléchargement… {done / 1e6:.0f} Mo sur {total / 1e6:.0f} Mo"
            )

    def _on_installed(self, path: str) -> None:
        if self._dialog is not None:
            self._dialog.close()
            self._dialog = None
        answer = QMessageBox.question(
            self.parent_widget,
            "Mise à jour installée",
            "La nouvelle version est installée. Redémarrer Chessio maintenant ?",
        )
        if answer == QMessageBox.StandardButton.Yes:
            os.execv(path, [path, *sys.argv[1:]])


def share(parent: QWidget) -> None:
    """Lien de téléchargement à partager, copié dans le presse-papiers."""
    QGuiApplication.clipboard().setText(RELEASES_PAGE)
    QMessageBox.information(
        parent,
        "Partager Chessio",
        "Lien de téléchargement copié dans le presse-papiers :\n\n"
        f"{RELEASES_PAGE}\n\n"
        "Il suffit de télécharger le fichier Chessio-….AppImage, de le rendre exécutable "
        "(clic droit → Propriétés → Autoriser l'exécution) puis de l'ouvrir. Chessio se "
        "met ensuite à jour lui-même (menu ⋮ → Rechercher une mise à jour).",
    )
