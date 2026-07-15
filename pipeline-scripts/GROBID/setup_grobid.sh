#!/usr/bin/env bash
# Build and start a local GROBID 0.8.1 server on port 8070.
# Debian/Ubuntu (including Google Colab). For most setups the Docker option
# in the README is simpler; use this if you want a local build.
set -euo pipefail

GROBID_VERSION="0.8.1"
GROBID_DIR="${GROBID_DIR:-$HOME/grobid}"

if ! command -v java >/dev/null 2>&1; then
    echo "Installing OpenJDK 17..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq openjdk-17-jdk-headless
fi

export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/java-17-openjdk-amd64}"

if [ ! -d "$GROBID_DIR" ]; then
    echo "Cloning GROBID $GROBID_VERSION into $GROBID_DIR..."
    git clone --depth 1 -b "$GROBID_VERSION" https://github.com/kermitt2/grobid "$GROBID_DIR"
fi

echo "Building GROBID (first build downloads the CRF models, takes a few minutes)..."
( cd "$GROBID_DIR" && ./gradlew clean install -q )

echo "Starting GROBID server on http://localhost:8070 ..."
nohup java -jar "$GROBID_DIR/grobid-service/build/libs/grobid-service-$GROBID_VERSION.jar" \
    server "$GROBID_DIR/grobid-service/config/config.yaml" > /tmp/grobid.log 2>&1 &

echo "Waiting for the server to come up (loads CRF models, 60-120s)..."
for _ in $(seq 1 120); do
    if [ "$(curl -s http://localhost:8070/api/isalive || true)" = "true" ]; then
        echo "GROBID is up."
        exit 0
    fi
    sleep 2
done

echo "Server did not respond in time. Last log lines:"
tail -40 /tmp/grobid.log
exit 1
