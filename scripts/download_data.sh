#!/usr/bin/env bash
# Download the dataset into data/ and verify it against known checksums.
#   data/<flight>/processed.csv, data/Flight Sheet.xlsx   authors' preprocessed flights (data_min.zip)
#   data/kilthub/parameters.csv, data/kilthub/README.txt   per-flight settings from KiltHub
#                                                           (Rodrigues et al. 2021, doi:10.1184/R1/12683453)
#   data/kilthub/flights/<flight>.csv                       raw per-flight logs from KiltHub (wind, GNSS, ...)
set -euo pipefail
cd "$(dirname "$0")/.."

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

curl -fsSL -o "$tmp/data_min.zip" "https://www.dropbox.com/s/vt5lbes7ojolto2/data_min.zip?dl=1"
curl -fsSL -o "$tmp/parameters.csv" "https://ndownloader.figshare.com/files/26385034"
curl -fsSL -o "$tmp/README.txt" "https://ndownloader.figshare.com/files/26405786"
curl -fsSL -o "$tmp/flights.zip" "https://ndownloader.figshare.com/files/26385070"

(cd "$tmp" && sha256sum -c --quiet - <<'EOF'
3bdc3c0c236f7b18dab64a642b59b64cf9c917b048c94149be05972b96ef6094  data_min.zip
568dffaed11a4d335ef4f8a932b342c836907af2f4aa54da9b4a0e2ffef2108a  parameters.csv
8601dd211d6c297c5411992dc34923bdb98b72cdc287517ed18e479f0e9540fd  README.txt
98f65a1ab54a8b618c169fdf49311a53ff1fefed86fdc3f440d6a0166ce49fb1  flights.zip
EOF
)

unzip -qo "$tmp/data_min.zip" -d . -x '__MACOSX/*'
mkdir -p data/kilthub
mv "$tmp/parameters.csv" "$tmp/README.txt" data/kilthub/
unzip -qo "$tmp/flights.zip" -d data/kilthub/

echo "done: $(ls data | grep -cE '^[0-9]+$') flights in data/, $(ls data/kilthub/flights | wc -l) raw logs and metadata in data/kilthub/"