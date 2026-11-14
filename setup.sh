#!/usr/bin/env bash
# ==============================================================================
# FLL-Vision Host Setup & Initialization Script (Zero Default Passwords)
# Creates .env, initializes the SQLite database & creates accounts:
#   - Admin Panel (/admin)                  : user 'admin' (Password REQUIRED)
#   - Volunteer Control (/control)          : optional user ('volunteers') + password
#   - System Configuration (/configuration) : optional user ('engineers') + password
#
# If parameters are not passed via CLI arguments, this script interactively prompts for them.
# ==============================================================================
set -e

ADMIN_USER=""
ADMIN_PASSWORD=""
VOLUNTEERS_USER=""
VOLUNTEERS_PASSWORD=""
ENGINEERS_USER=""
ENGINEERS_PASSWORD=""
PORT="8080"
START_DOCKER=0
NON_INTERACTIVE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --admin-user|--master-user)
            ADMIN_USER="$2"; shift 2 ;;
        --admin-password|--master-password)
            ADMIN_PASSWORD="$2"; shift 2 ;;
        --volunteers-user|--volunteer-user)
            VOLUNTEERS_USER="$2"; shift 2 ;;
        --volunteers-password|--volunteer-password)
            VOLUNTEERS_PASSWORD="$2"; shift 2 ;;
        --engineers-user|--backend-user)
            ENGINEERS_USER="$2"; shift 2 ;;
        --engineers-password|--backend-password)
            ENGINEERS_PASSWORD="$2"; shift 2 ;;
        --port)
            PORT="$2"; shift 2 ;;
        --start-docker)
            START_DOCKER=1; shift 1 ;;
        --non-interactive)
            NON_INTERACTIVE=1; shift 1 ;;
        *)
            echo "Unknown parameter: $1"
            echo "Usage: ./setup.sh [--admin-user admin] [--admin-password <pw>] [--volunteers-user volunteers --volunteers-password <pw>] [--engineers-user engineers --engineers-password <pw>] [--port 8080] [--start-docker]"
            exit 1 ;;
    esac
done

# 1. Interactive prompt for Admin User & Password if not passed as arguments
if [ -z "$ADMIN_USER" ]; then
    if [ "$NON_INTERACTIVE" -eq 0 ] && [ -z "$ADMIN_PASSWORD" ] && [ -t 0 ]; then
        read -r -p "Admin-Benutzername für Admin Panel (/admin) [Standard: admin]: " INPUT_ADMIN_USER
        ADMIN_USER="${INPUT_ADMIN_USER:-admin}"
    else
        ADMIN_USER="admin"
    fi
fi

while [ -z "$ADMIN_PASSWORD" ] || [ "${#ADMIN_PASSWORD}" -lt 4 ]; do
    if [ "$NON_INTERACTIVE" -eq 1 ] || [ ! -t 0 ]; then
        echo "Error: --admin-password (mindestens 4 Zeichen) ist erforderlich!"
        exit 1
    fi
    read -r -s -p "Admin-Passwort für '${ADMIN_USER}' (/admin, ERFORDERLICH, mind. 4 Zeichen): " ADMIN_PASSWORD
    echo ""
    if [ "${#ADMIN_PASSWORD}" -lt 4 ]; then
        echo "  [!] Bitte ein Passwort mit mindestens 4 Zeichen eingeben."
    fi
done

# 2. Interactive prompt for optional Volunteer Control (/control) account
if [ -z "$VOLUNTEERS_PASSWORD" ] && [ "$NON_INTERACTIVE" -eq 0 ] && [ -t 0 ]; then
    echo ""
    echo "--- Optional: Benutzer für Volunteer Control (/control) anlegen ---"
    read -r -s -p "Optionales Passwort für Volunteer Control (/control) [Enter zum Überspringen]: " VOLUNTEERS_PASSWORD
    echo ""
    if [ -n "$VOLUNTEERS_PASSWORD" ]; then
        if [ -z "$VOLUNTEERS_USER" ]; then
            read -r -p "Benutzername für Volunteer Control (/control) [Standard: volunteers]: " INPUT_VOL_USER
            VOLUNTEERS_USER="${INPUT_VOL_USER:-volunteers}"
        fi
    fi
elif [ -n "$VOLUNTEERS_PASSWORD" ] && [ -z "$VOLUNTEERS_USER" ]; then
    VOLUNTEERS_USER="volunteers"
fi

# 3. Interactive prompt for optional System Configuration (/configuration) account
if [ -z "$ENGINEERS_PASSWORD" ] && [ "$NON_INTERACTIVE" -eq 0 ] && [ -t 0 ]; then
    echo ""
    echo "--- Optional: Benutzer für System Configuration (/configuration) anlegen ---"
    read -r -s -p "Optionales Passwort für System Configuration (/configuration) [Enter zum Überspringen]: " ENGINEERS_PASSWORD
    echo ""
    if [ -n "$ENGINEERS_PASSWORD" ]; then
        if [ -z "$ENGINEERS_USER" ]; then
            read -r -p "Benutzername für System Configuration (/configuration) [Standard: engineers]: " INPUT_ENG_USER
            ENGINEERS_USER="${INPUT_ENG_USER:-engineers}"
        fi
    fi
elif [ -n "$ENGINEERS_PASSWORD" ] && [ -z "$ENGINEERS_USER" ]; then
    ENGINEERS_USER="engineers"
fi

mkdir -p data/training_videos data/videos data/outputs data/db models

cat > .env <<EOF
FLL_PORT=${PORT}
FLL_ADMIN_USER=${ADMIN_USER}
FLL_ADMIN_PASSWORD=${ADMIN_PASSWORD}
FLL_VOLUNTEERS_USER=${VOLUNTEERS_USER}
FLL_VOLUNTEERS_PASSWORD=${VOLUNTEERS_PASSWORD}
FLL_ENGINEERS_USER=${ENGINEERS_USER}
FLL_ENGINEERS_PASSWORD=${ENGINEERS_PASSWORD}
EOF
chmod 600 .env
echo "[OK] Written environment configuration to .env"

if command -v python3 >/dev/null 2>&1; then
    CMD=(python3 scripts/setup_db.py --non-interactive --db-path "data/db/fll_vision.sqlite3" --admin-user "$ADMIN_USER" --admin-password "$ADMIN_PASSWORD")
    if [ -n "$VOLUNTEERS_USER" ] && [ -n "$VOLUNTEERS_PASSWORD" ]; then
        CMD+=(--volunteers-user "$VOLUNTEERS_USER" --volunteers-password "$VOLUNTEERS_PASSWORD")
    fi
    if [ -n "$ENGINEERS_USER" ] && [ -n "$ENGINEERS_PASSWORD" ]; then
        CMD+=(--engineers-user "$ENGINEERS_USER" --engineers-password "$ENGINEERS_PASSWORD")
    fi
    "${CMD[@]}"
fi

if [ "$START_DOCKER" -eq 1 ]; then
    echo "[Docker] Building and starting single-container FLL-Vision system..."
    docker compose up --build -d
    echo "[OK] FLL-Vision is running on http://localhost:${PORT}"
else
    echo "[Next Step] Run 'docker compose up --build -d' to start the single container."
fi
