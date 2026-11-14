"""
Database & Authentication Layer for FLL-Vision.
Implements SQLite storage with PBKDF2-HMAC-SHA256 (600,000 iterations, OWASP 2026 standard)
password hashing, brute-force rate limiting, comprehensive error handling/logging, and
strict role-based access control (RBAC) for the 3 protected areas:
  - 'volunteers' : Volunteer Control (/control, login: /login/control)
  - 'engineers'  : System Configuration (/configuration, login: /login/configuration)
  - 'admin'      : Admin Panel (/admin, login: /login/admin + access to all protected areas)
"""

import hashlib
import hmac
import json
import logging
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("fll-vision.database")

VALID_ROLES = ("volunteers", "engineers", "admin")

# Map legacy role aliases gracefully to the canonical role names
ROLE_ALIASES = {
	"volunteers": "volunteers",
	"volunteer": "volunteers",
	"control": "volunteers",
	"engineers": "engineers",
	"engineer": "engineers",
	"backend_admin": "engineers",
	"backend": "engineers",
	"configuration": "engineers",
	"admin": "admin",
	"master": "admin",
}


def normalize_role(role: str) -> str:
	norm = ROLE_ALIASES.get(str(role).strip().lower())
	if not norm or norm not in VALID_ROLES:
		raise ValueError(f"Invalid role '{role}'. Must be one of {VALID_ROLES}.")
	return norm


def hash_password(password: str, salt: Optional[bytes] = None) -> str:
	"""Hashes a password using PBKDF2-HMAC-SHA256 with 600,000 iterations and a 16-byte random salt."""
	if not password or len(password) < 4:
		raise ValueError("Password must be at least 4 characters long.")
	if salt is None:
		salt = secrets.token_bytes(16)
	iterations = 600_000
	dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
	return f"pbkdf2_sha256${iterations}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
	"""Verifies a plaintext password against a stored PBKDF2-HMAC-SHA256 hash in constant time."""
	try:
		algo, iter_str, salt_hex, hash_hex = stored_hash.split("$")
		if algo != "pbkdf2_sha256":
			return False
		iterations = int(iter_str)
		salt = bytes.fromhex(salt_hex)
		expected = bytes.fromhex(hash_hex)
		computed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
		return hmac.compare_digest(computed, expected)
	except Exception as exc:
		logger.warning(f"Password verification error: {exc}")
		return False


class FLLDatabase:
	"""Thread-safe SQLite database manager for users (volunteers, engineers, admin), sessions, teams, and runs."""

	def __init__(self, db_path: str = "/workspace/data/db/fll_vision.sqlite3"):
		self.db_path = Path(db_path)
		self.db_path.parent.mkdir(parents=True, exist_ok=True)
		self._lock = threading.Lock()
		self.init_schema()

	def _connect(self) -> sqlite3.Connection:
		conn = sqlite3.connect(str(self.db_path), timeout=15.0, check_same_thread=False)
		conn.row_factory = sqlite3.Row
		conn.execute("PRAGMA foreign_keys = ON;")
		conn.execute("PRAGMA journal_mode = WAL;")
		conn.execute("PRAGMA synchronous = NORMAL;")
		return conn

	def init_schema(self):
		try:
			with self._lock, self._connect() as conn:
				conn.executescript(
					"""
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT UNIQUE NOT NULL,
                        password_hash TEXT NOT NULL,
                        role TEXT NOT NULL CHECK(role IN ('volunteers', 'engineers', 'admin')),
                        created_at REAL NOT NULL,
                        last_login REAL
                    );

                    CREATE TABLE IF NOT EXISTS sessions (
                        token TEXT PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        username TEXT NOT NULL,
                        role TEXT NOT NULL,
                        csrf_token TEXT NOT NULL,
                        expires_at REAL NOT NULL,
                        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS login_attempts (
                        ip_address TEXT PRIMARY KEY,
                        failed_count INTEGER NOT NULL DEFAULT 0,
                        locked_until REAL NOT NULL DEFAULT 0
                    );

                    CREATE TABLE IF NOT EXISTS teams (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        team_number TEXT NOT NULL,
                        team_name TEXT NOT NULL,
                        organization TEXT DEFAULT ''
                    );

                    CREATE TABLE IF NOT EXISTS matches (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        round_name TEXT NOT NULL,
                        team_left_name TEXT NOT NULL,
                        team_right_name TEXT DEFAULT '',
                        duration_sec INTEGER NOT NULL DEFAULT 150,
                        status TEXT NOT NULL DEFAULT 'idle',
                        started_at REAL,
                        ended_at REAL
                    );

                    CREATE TABLE IF NOT EXISTS runs_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        match_id INTEGER,
                        run_number INTEGER NOT NULL,
                        duration_s REAL NOT NULL,
                        distance_mm REAL NOT NULL,
                        avg_speed_mm_s REAL NOT NULL,
                        interrupted_by_hand INTEGER NOT NULL DEFAULT 0,
                        trajectory_json TEXT NOT NULL,
                        created_at REAL NOT NULL
                    );
                    """
				)
				cur = conn.execute("SELECT COUNT(*) as cnt FROM teams")
				if cur.fetchone()["cnt"] == 0:
					default_teams = [
						("101", "BIOGLOW Builders", "HTL Hollabrunn"),
						("102", "RoboSpark Prime", "Gymnasium Hollabrunn"),
						("103", "CyberBricks", "HTL St. Pölten"),
						("104", "Quantum Gears", "TU Wien Junior"),
						("105", "Neon Navigators", "HTL Krems"),
						("106", "Alpha Centauri", "BG/BRG Tulln"),
					]
					conn.executemany(
						"INSERT INTO teams (team_number, team_name, organization) VALUES (?, ?, ?)",
						default_teams,
					)
				conn.commit()
		except sqlite3.Error as exc:
			logger.error(f"Failed to initialize SQLite schema at {self.db_path}: {exc}")
			raise

	# --------------------------------------------------------------------------
	# User Management ('volunteers', 'engineers', 'admin')
	# --------------------------------------------------------------------------
	def create_or_update_user(self, username: str, password: str, role: str) -> Dict:
		username = username.strip()
		if not username:
			raise ValueError("Username cannot be empty.")
		canonical_role = normalize_role(role)

		pw_hash = hash_password(password)
		now = time.time()

		with self._lock, self._connect() as conn:
			existing = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
			if existing:
				conn.execute(
					"UPDATE users SET password_hash = ?, role = ? WHERE username = ?",
					(pw_hash, canonical_role, username),
				)
				user_id = existing["id"]
				logger.info(f"Updated user '{username}' with role '{canonical_role}'.")
			else:
				cur = conn.execute(
					"INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
					(username, pw_hash, canonical_role, now),
				)
				user_id = cur.lastrowid
				logger.info(f"Created user '{username}' with role '{canonical_role}'.")
			conn.commit()
		return {"id": user_id, "username": username, "role": canonical_role}

	def list_users(self) -> List[Dict]:
		with self._lock, self._connect() as conn:
			rows = conn.execute(
				"SELECT id, username, role, created_at, last_login FROM users ORDER BY id ASC"
			).fetchall()
			return [dict(r) for r in rows]

	def delete_user(self, user_id: int) -> bool:
		with self._lock, self._connect() as conn:
			row = conn.execute("SELECT username, role FROM users WHERE id = ?", (user_id,)).fetchone()
			if not row:
				return False
			if row["role"] == "admin":
				admins = conn.execute("SELECT COUNT(*) as cnt FROM users WHERE role = 'admin'").fetchone()["cnt"]
				if admins <= 1:
					raise ValueError("Der letzte verbleibende 'admin'-Benutzer kann nicht gelöscht werden.")
			conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
			conn.commit()
			logger.info(f"Deleted user id={user_id} ('{row['username']}').")
			return True

	# --------------------------------------------------------------------------
	# Authentication, Brute-Force Protection & Session Management
	# --------------------------------------------------------------------------
	def check_rate_limit(self, ip_address: str) -> Tuple[bool, int]:
		"""Returns (allowed, retry_after_seconds). Locks IP for 60s after 8 failed logins."""
		now = time.time()
		with self._lock, self._connect() as conn:
			row = conn.execute(
				"SELECT failed_count, locked_until FROM login_attempts WHERE ip_address = ?",
				(ip_address,),
			).fetchone()
			if row and row["locked_until"] > now:
				return False, int(row["locked_until"] - now) + 1
			return True, 0

	def _record_login_result(self, conn: sqlite3.Connection, ip_address: str, success: bool):
		if success:
			conn.execute("DELETE FROM login_attempts WHERE ip_address = ?", (ip_address,))
			return
		row = conn.execute("SELECT failed_count FROM login_attempts WHERE ip_address = ?", (ip_address,)).fetchone()
		count = (row["failed_count"] + 1) if row else 1
		locked_until = (time.time() + 60.0) if count >= 8 else 0.0
		conn.execute(
			"""
            INSERT INTO login_attempts (ip_address, failed_count, locked_until)
            VALUES (?, ?, ?)
            ON CONFLICT(ip_address) DO UPDATE SET failed_count = excluded.failed_count, locked_until = excluded.locked_until
            """,
			(ip_address, count, locked_until),
		)

	@staticmethod
	def _allowed_roles_for_area(required_area: str) -> Tuple[str, ...]:
		canonical = normalize_role(required_area)
		return {
			"volunteers": ("volunteers", "admin"),
			"engineers": ("engineers", "admin"),
			"admin": ("admin",),
		}[canonical]

	def authenticate_user(
		self,
		username: str,
		password: str,
		required_area: str,
		ip_address: str = "127.0.0.1",
		session_ttl_sec: int = 43200,
	) -> Optional[Dict]:
		"""
		Authenticates a user for a specific protected area:
		  - required_area == 'volunteers' (/control)       -> allows ('volunteers', 'admin')
		  - required_area == 'engineers'  (/configuration) -> allows ('engineers', 'admin')
		  - required_area == 'admin'      (/admin)         -> allows ('admin',)
		"""
		allowed, _ = self.check_rate_limit(ip_address)
		if not allowed:
			logger.warning(f"Rate-limited login attempt from IP {ip_address} for user '{username}'.")
			return None

		allowed_roles = self._allowed_roles_for_area(required_area)

		with self._lock, self._connect() as conn:
			row = conn.execute(
				"SELECT id, username, password_hash, role FROM users WHERE username = ?",
				(username.strip(),),
			).fetchone()

			if not row or not verify_password(password, row["password_hash"]):
				self._record_login_result(conn, ip_address, success=False)
				conn.commit()
				logger.warning(f"Failed login for username='{username}' from IP {ip_address}.")
				return None

			if row["role"] not in allowed_roles:
				logger.warning(
					f"Unauthorized area access: user '{username}' (role='{row['role']}') tried to access area '{required_area}'."
				)
				return None

			self._record_login_result(conn, ip_address, success=True)
			now = time.time()
			token = secrets.token_urlsafe(32)
			csrf_token = secrets.token_hex(16)
			expires_at = now + session_ttl_sec

			conn.execute("UPDATE users SET last_login = ? WHERE id = ?", (now, row["id"]))
			conn.execute(
				"INSERT INTO sessions (token, user_id, username, role, csrf_token, expires_at) VALUES (?, ?, ?, ?, ?, ?)",
				(token, row["id"], row["username"], row["role"], csrf_token, expires_at),
			)
			conn.commit()

			logger.info(
				f"Successful login: user='{row['username']}' (role='{row['role']}') for area='{required_area}'."
			)
			return {
				"token": token,
				"csrf_token": csrf_token,
				"user_id": row["id"],
				"username": row["username"],
				"role": row["role"],
				"expires_at": expires_at,
			}

	def validate_session(self, token: Optional[str], required_area: str) -> Optional[Dict]:
		if not token:
			return None
		allowed_roles = self._allowed_roles_for_area(required_area)

		now = time.time()
		with self._lock, self._connect() as conn:
			row = conn.execute(
				"SELECT token, user_id, username, role, csrf_token, expires_at FROM sessions WHERE token = ?",
				(token,),
			).fetchone()
			if not row:
				return None
			if row["expires_at"] < now:
				conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
				conn.commit()
				return None
			if row["role"] not in allowed_roles:
				return None
			return dict(row)

	def logout_session(self, token: Optional[str]):
		if not token:
			return
		with self._lock, self._connect() as conn:
			conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
			conn.commit()

	# --------------------------------------------------------------------------
	# Teams & Match Runs Persistence
	# --------------------------------------------------------------------------
	def list_teams(self) -> List[Dict]:
		with self._lock, self._connect() as conn:
			rows = conn.execute(
				"SELECT id, team_number, team_name, organization FROM teams ORDER BY team_number ASC"
			).fetchall()
			return [dict(r) for r in rows]

	def add_team(self, team_number: str, team_name: str, organization: str = "") -> Dict:
		num = team_number.strip()
		name = team_name.strip()
		if not name:
			raise ValueError("Team name cannot be empty.")
		with self._lock, self._connect() as conn:
			cur = conn.execute(
				"INSERT INTO teams (team_number, team_name, organization) VALUES (?, ?, ?)",
				(num, name, organization.strip()),
			)
			conn.commit()
			return {"id": cur.lastrowid, "team_number": num, "team_name": name, "organization": organization.strip()}

	def record_completed_run(
		self,
		run_number: int,
		duration_s: float,
		distance_mm: float,
		avg_speed_mm_s: float,
		interrupted_by_hand: bool,
		trajectory: List[Dict],
		match_id: Optional[int] = None,
	):
		with self._lock, self._connect() as conn:
			conn.execute(
				"""
                INSERT INTO runs_log
                (match_id, run_number, duration_s, distance_mm, avg_speed_mm_s, interrupted_by_hand, trajectory_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
				(
					match_id,
					run_number,
					round(duration_s, 2),
					round(distance_mm, 1),
					round(avg_speed_mm_s, 1),
					1 if interrupted_by_hand else 0,
					json.dumps(trajectory),
					time.time(),
				),
			)
			conn.commit()

	def get_recorded_runs(self, limit: int = 50) -> List[Dict]:
		with self._lock, self._connect() as conn:
			rows = conn.execute(
				"SELECT id, match_id, run_number, duration_s, distance_mm, avg_speed_mm_s, interrupted_by_hand, created_at FROM runs_log ORDER BY id DESC LIMIT ?",
				(limit,),
			).fetchall()
			return [dict(r) for r in rows]
