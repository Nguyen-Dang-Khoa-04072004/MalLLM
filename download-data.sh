#!/bin/bash

# Fixed Git repository URLs
BENIGN_URL="https://github.com/hogiathang/benign-datasets"
MALICIOUS_URL="https://github.com/hogiathang/malware-dataset"

# Create target directories
mkdir -p data/benign
mkdir -p data/malicious

# Function to download repo directly into a folder without extra subfolder
clone_into() {
    local url="$1"
    local target_dir="$2"

    # Temporary folder
    tmp_dir=$(mktemp -d)

    # Clone repo into temporary folder
    git clone --depth 1 "$url" "$tmp_dir"

    # Copy contents into target folder (overwrite)
    cp -r "$tmp_dir"/* "$target_dir"/

    # Clean up
    rm -rf "$tmp_dir"
}

echo "Downloading benign repository..."
clone_into "$BENIGN_URL" "data/benign"

echo "Downloading malicious repository..."
clone_into "$MALICIOUS_URL" "data/malicious"

echo "Done!"
echo "Benign repo contents stored in: data/benign/"
echo "Malicious repo contents stored in: data/malicious/"


