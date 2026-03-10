#!/usr/bin/env bash
set -euo pipefail

# Download MaxQuant proteinGroups.txt from PRIDE (PXD046440)
# Destination: data/MaxQuant/proteinGroups.txt

URL="ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2023/12/PXD046440/proteinGroups.txt"
DEST_DIR="$(dirname "$0")/data/MaxQuant"
DEST_FILE="${DEST_DIR}/proteinGroups.txt"

mkdir -p "${DEST_DIR}"

echo "Downloading proteinGroups.txt to ${DEST_FILE}"
echo "Source: ${URL}"

# Prefer curl if available; fallback to wget
if command -v curl >/dev/null 2>&1; then
  # -f: fail on HTTP/FTP errors
  # -L: follow redirects (harmless for FTP)
  # --retry/--retry-delay: robustness
  # -C -: resume partial downloads
  curl -fL --retry 3 --retry-delay 5 -C - -o "${DEST_FILE}" "${URL}"
elif command -v wget >/dev/null 2>&1; then
  # -c: continue/resume
  # --tries: robustness
  wget -c --tries=3 -O "${DEST_FILE}" "${URL}"
else
  echo "Error: Neither curl nor wget is installed. Please install one and retry." >&2
  exit 1
fi

echo "Download complete: ${DEST_FILE}"


