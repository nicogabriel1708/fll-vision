#!/usr/bin/env python3
"""
FLL-Vision Database & User Initialization Script.
Creates the SQLite database, initializes all tables, and sets up user accounts:
  - Admin Panel (/admin)                  : required password for user 'admin' (role='admin')
  - Volunteer Control (/control)          : optional user (default 'volunteers', role='volunteers') + password
  - System Configuration (/configuration) : optional user (default 'engineers', role='engineers') + password

If arguments are not passed on the command line, the script interactively prompts for them.
No default passwords exist anywhere in the codebase.
"""
import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import FLLDatabase


def prompt_secret(prompt_text: str, required: bool = False) -> str:
    while True:
        try:
            val = getpass.getpass(prompt_text).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAbgebrochen.")
            sys.exit(1)
        if not val and not required:
            return ""
        if len(val) < 4:
            print("  [!] Das Passwort muss mindestens 4 Zeichen lang sein.")
            continue
        return val


def main():
    parser = argparse.ArgumentParser(
        description="Initialize FLL-Vision SQLite Database (admin password required; volunteers & engineers optional)."
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="/workspace/data/db/fll_vision.sqlite3",
        help="Path to SQLite database file",
    )
    parser.add_argument(
        "--admin-user", "--master-user",
        dest="admin_user",
        type=str,
        default=None,
        help="Admin Panel (/admin) username (default: admin)",
    )
    parser.add_argument(
        "--admin-password", "--master-password",
        dest="admin_password",
        type=str,
        default=None,
        help="Admin Panel (/admin) password (REQUIRED; prompted interactively if omitted)",
    )
    parser.add_argument(
        "--volunteers-user", "--volunteer-user",
        dest="volunteers_user",
        type=str,
        default=None,
        help="Optional Volunteer Control (/control) username (default: volunteers)",
    )
    parser.add_argument(
        "--volunteers-password", "--volunteer-password",
        dest="volunteers_password",
        type=str,
        default=None,
        help="Optional password for Volunteer Control (/control) user",
    )
    parser.add_argument(
        "--engineers-user", "--backend-user",
        dest="engineers_user",
        type=str,
        default=None,
        help="Optional System Configuration (/configuration) username (default: engineers)",
    )
    parser.add_argument(
        "--engineers-password", "--backend-password",
        dest="engineers_password",
        type=str,
        default=None,
        help="Optional password for System Configuration (/configuration) user",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Do not prompt interactively for optional accounts if omitted",
    )

    args = parser.parse_args()
    is_interactive = sys.stdin.isatty() and not args.non_interactive

    # 1. Admin User & Password (REQUIRED)
    admin_user = args.admin_user
    if not admin_user:
        if is_interactive and args.admin_password is None:
            entered = input("Admin-Benutzername für Admin Panel (/admin) [Standard: admin]: ").strip()
            admin_user = entered if entered else "admin"
        else:
            admin_user = "admin"

    admin_password = args.admin_password
    if not admin_password:
        if sys.stdin.isatty():
            admin_password = prompt_secret(
                f"Admin-Passwort für '{admin_user}' (/admin, ERFORDERLICH): ", required=True
            )
        else:
            parser.error("--admin-password ist erforderlich, wenn kein interaktives Terminal aktiv ist.")

    # 2. Optional Volunteers User & Password (/control)
    volunteers_user = args.volunteers_user
    volunteers_password = args.volunteers_password
    if volunteers_password is None and is_interactive:
        print("\n--- Optional: Volunteer Control (/control) Benutzer anlegen ---")
        vol_u = input("Benutzername für Volunteer Control (/control) [Enter für 'volunteers', oder '-' zum Überspringen]: ").strip()
        if vol_u != "-":
            volunteers_user = vol_u if vol_u else "volunteers"
            volunteers_password = prompt_secret(
                f"Optionales Passwort für '{volunteers_user}' (/control) [Enter zum Überspringen]: ",
                required=False,
            )
    elif volunteers_password and not volunteers_user:
        volunteers_user = "volunteers"

    # 3. Optional Engineers User & Password (/configuration)
    engineers_user = args.engineers_user
    engineers_password = args.engineers_password
    if engineers_password is None and is_interactive:
        print("\n--- Optional: System Configuration (/configuration) Benutzer anlegen ---")
        eng_u = input("Benutzername für System Configuration (/configuration) [Enter für 'engineers', oder '-' zum Überspringen]: ").strip()
        if eng_u != "-":
            engineers_user = eng_u if eng_u else "engineers"
            engineers_password = prompt_secret(
                f"Optionales Passwort für '{engineers_user}' (/configuration) [Enter zum Überspringen]: ",
                required=False,
            )
    elif engineers_password and not engineers_user:
        engineers_user = "engineers"

    db = FLLDatabase(db_path=args.db_path)
    adm = db.create_or_update_user(admin_user, admin_password, role="admin")
    print(f"[OK] Database initialized at: {args.db_path}")
    print(f"[OK] Admin Panel (/admin) user created/updated: username='{adm['username']}' (role='admin')")

    if volunteers_user and volunteers_password:
        vol = db.create_or_update_user(volunteers_user, volunteers_password, role="volunteers")
        print(f"[OK] Volunteer Control (/control) user created/updated: username='{vol['username']}' (role='volunteers')")

    if engineers_user and engineers_password:
        eng = db.create_or_update_user(engineers_user, engineers_password, role="engineers")
        print(f"[OK] System Configuration (/configuration) user created/updated: username='{eng['username']}' (role='engineers')")


if __name__ == "__main__":
    main()
