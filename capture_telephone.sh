#!/bin/bash

DOSSIER="$HOME/Captures_Telephone"

mkdir -p "$DOSSIER"

# Vérifier que ADB est installé
if ! command -v adb &> /dev/null; then
    echo "❌ ADB n'est pas installé."
    echo "Installe-le avec : sudo apt install adb"
    exit 1
fi

# Vérifier qu'un téléphone est connecté
if ! adb get-state &> /dev/null; then
    echo "❌ Aucun téléphone Android connecté ou autorisé."
    echo "Vérifie le câble et active le débogage USB."
    exit 1
fi

echo "======================================"
echo "📱 Téléphone connecté"
echo "Appuie sur ENTRÉE pour faire une capture"
echo "Tape q puis ENTRÉE pour quitter"
echo "======================================"

while true; do
    read -r -p "> " choix

    if [ "$choix" = "q" ]; then
        echo "Fin du programme."
        break
    fi

    DATE=$(date +"%Y-%m-%d_%H-%M-%S")
    FICHIER="$DOSSIER/capture_$DATE.png"

    echo "📸 Capture en cours..."

    if adb shell screencap -p /sdcard/capture_temp.png && \
       adb pull /sdcard/capture_temp.png "$FICHIER"; then

        adb shell rm /sdcard/capture_temp.png

        echo "✅ Capture enregistrée : $FICHIER"
    else
        echo "❌ Erreur lors de la capture."
    fi
done
