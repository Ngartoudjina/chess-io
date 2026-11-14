"""Capture de l'écran de l'ordinateur par le portail XDG (seule voie permise sous Wayland).

L'appel est asynchrone : le portail répond par un signal ``Response`` sur l'objet
« requête » qu'il crée. La première fois, GNOME demande à l'utilisateur
d'autoriser Chessio à faire des captures d'écran.
"""

from __future__ import annotations

from urllib.parse import unquote, urlparse

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtDBus import QDBusConnection, QDBusInterface

from chessio.vision.board import load_image

PORTAL = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"


class ScreenCapturer(QObject):
    captured = Signal(object)  # image RGB (numpy)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._bus = QDBusConnection.sessionBus()
        self._counter = 0
        self._pending: str | None = None

    def request(self) -> None:
        """Demande une capture de tout l'écran (réponse : ``captured`` ou ``failed``)."""
        if not self._bus.isConnected():
            self.failed.emit("Bus D-Bus de session indisponible : capture d'écran impossible.")
            return
        self._counter += 1
        token = f"chessio{self._counter}"
        sender = self._bus.baseService().lstrip(":").replace(".", "_")
        path = f"{PORTAL_PATH}/request/{sender}/{token}"
        if self._pending is not None:
            self._disconnect(self._pending)
        self._pending = path
        self._bus.connect(
            PORTAL,
            path,
            "org.freedesktop.portal.Request",
            "Response",
            self,
            "1_on_response(uint,QVariantMap)",
        )
        interface = QDBusInterface(
            PORTAL, PORTAL_PATH, "org.freedesktop.portal.Screenshot", self._bus
        )
        reply = interface.call("Screenshot", "", {"handle_token": token, "interactive": False})
        if reply.errorName():
            self.failed.emit(f"Capture d'écran refusée : {reply.errorMessage()}")

    def _disconnect(self, path: str) -> None:
        self._bus.disconnect(
            PORTAL,
            path,
            "org.freedesktop.portal.Request",
            "Response",
            self,
            "1_on_response(uint,QVariantMap)",
        )

    @Slot("uint", "QVariantMap")
    def _on_response(self, response: int, results: dict) -> None:
        if self._pending is not None:
            self._disconnect(self._pending)
            self._pending = None
        if response != 0:
            self.failed.emit("Capture d'écran annulée ou refusée.")
            return
        uri = str(results.get("uri", ""))
        path = unquote(urlparse(uri).path)
        try:
            self.captured.emit(load_image(path))
        except OSError as exc:
            self.failed.emit(f"Capture d'écran illisible : {exc}")
