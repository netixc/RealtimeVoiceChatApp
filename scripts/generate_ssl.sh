#!/bin/bash

# Script to generate SSL certificates if not present

SSL_CERT_PATH="${SSL_CERT_PATH:-/app/certs/cert.pem}"
SSL_KEY_PATH="${SSL_KEY_PATH:-/app/certs/key.pem}"
CERT_DIR="$(dirname "$SSL_CERT_PATH")"

echo "🔐 Checking SSL certificate setup..."

# Create certificates directory if it doesn't exist
mkdir -p "$CERT_DIR"

# Check if SSL certificates exist
if [[ -f "$SSL_CERT_PATH" && -f "$SSL_KEY_PATH" ]]; then
    echo "✅ SSL certificates already exist:"
    echo "   Certificate: $SSL_CERT_PATH"
    echo "   Key: $SSL_KEY_PATH"
else
    echo "🔧 Generating self-signed SSL certificates..."
    
    # Generate self-signed certificate
    openssl req -x509 -newkey rsa:4096 -keyout "$SSL_KEY_PATH" -out "$SSL_CERT_PATH" -days 365 -nodes \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost" \
        -addext "subjectAltName=DNS:localhost,DNS:*.localhost,IP:127.0.0.1,IP:0.0.0.0"
    
    if [[ $? -eq 0 ]]; then
        echo "✅ SSL certificates generated successfully:"
        echo "   Certificate: $SSL_CERT_PATH"
        echo "   Key: $SSL_KEY_PATH"
        
        # Set proper permissions
        chmod 600 "$SSL_KEY_PATH"
        chmod 644 "$SSL_CERT_PATH"
        
        # Change ownership to appuser if we're running as root
        if [[ $(id -u) -eq 0 ]]; then
            chown appuser:appgroup "$SSL_CERT_PATH" "$SSL_KEY_PATH" 2>/dev/null || true
        fi
    else
        echo "❌ Failed to generate SSL certificates"
        exit 1
    fi
fi

echo "🔐 SSL certificate setup complete."