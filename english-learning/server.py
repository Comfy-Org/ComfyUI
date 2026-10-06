#!/usr/bin/env python3
"""Serve Daylight and its single-user, self-hosted account API."""

import hashlib
import hmac
import http.cookies
import json
import mimetypes
import os
import re
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parent
DATABASE = Path(os.environ.get("DAYLIGHT_DATABASE", ROOT / "daylight.sqlite3")).expanduser()
SESSION_SECONDS = 30 * 24 * 60 * 60
PASSWORD_ITERATIONS = 600_000
MAX_BODY_BYTES = 512 * 1024
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
SETUP_KEY = secrets.token_urlsafe(24)
SECURE_COOKIE = os.environ.get("DAYLIGHT_SECURE_COOKIE", "0") == "1"
LOGIN_FAILURES = {}
LOGIN_FAILURES_LOCK = threading.Lock()


def connect_database():
    DATABASE.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DATABASE, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def database_connection():
    connection = connect_database()
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database():
    with database_connection() as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_salt BLOB NOT NULL,
                password_hash BLOB NOT NULL,
                created_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS sessions_expiry ON sessions(expires_at);
            CREATE TABLE IF NOT EXISTS learning_data (
                user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
                data_json TEXT NOT NULL,
                revision INTEGER NOT NULL DEFAULT 0,
                updated_at INTEGER NOT NULL
            );
        """)
    if DATABASE.exists():
        DATABASE.chmod(0o600)


def derive_password(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)


def session_token_hash(token):
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def registration_open():
    with database_connection() as connection:
        return connection.execute("SELECT 1 FROM users LIMIT 1").fetchone() is None


class DaylightHandler(SimpleHTTPRequestHandler):
    server_version = "Daylight"
    sys_version = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'",
        )
        super().end_headers()

    def send_json(self, status, payload, cookie=None, clear_cookie=False):
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if cookie:
            self.send_header("Set-Cookie", self.session_cookie(cookie))
        elif clear_cookie:
            self.send_header("Set-Cookie", self.session_cookie("", clear=True))
        self.end_headers()
        self.wfile.write(body)

    def session_cookie(self, token, clear=False):
        cookie = http.cookies.SimpleCookie()
        cookie["daylight_session"] = token
        morsel = cookie["daylight_session"]
        morsel["path"] = "/"
        morsel["httponly"] = True
        morsel["samesite"] = "Strict"
        morsel["max-age"] = 0 if clear else SESSION_SECONDS
        if SECURE_COOKIE:
            morsel["secure"] = True
        return morsel.OutputString()

    def read_json(self):
        raw_length = self.headers.get("Content-Length")
        if raw_length is None or not raw_length.isdecimal():
            raise ValueError("A valid Content-Length is required.")
        length = int(raw_length)
        if length > MAX_BODY_BYTES:
            raise OverflowError("Request body is too large.")
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("A JSON object is required.")
        return payload

    def has_same_origin(self):
        origin = self.headers.get("Origin")
        host = self.headers.get("Host")
        if not origin or not host:
            return False
        parsed = urlsplit(origin)
        return (
            parsed.scheme in ("http", "https")
            and parsed.netloc.lower() == host.lower()
            and parsed.path in ("", "/")
            and not parsed.query
            and not parsed.fragment
        )

    def current_user(self, connection):
        cookies = http.cookies.SimpleCookie()
        try:
            cookies.load(self.headers.get("Cookie", ""))
        except http.cookies.CookieError:
            return None
        morsel = cookies.get("daylight_session")
        if not morsel or not morsel.value:
            return None
        row = connection.execute(
            "SELECT users.id, users.username, sessions.expires_at "
            "FROM sessions JOIN users ON users.id = sessions.user_id WHERE sessions.token_hash = ?",
            (session_token_hash(morsel.value),),
        ).fetchone()
        if row is None:
            return None
        if row["expires_at"] <= int(time.time()):
            connection.execute("DELETE FROM sessions WHERE token_hash = ?", (session_token_hash(morsel.value),))
            return None
        return row

    def set_session(self, connection, user_id):
        token = secrets.token_urlsafe(32)
        connection.execute("DELETE FROM sessions WHERE expires_at <= ?", (int(time.time()),))
        connection.execute(
            "INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
            (session_token_hash(token), user_id, int(time.time()) + SESSION_SECONDS),
        )
        return token

    def fail(self, status, message):
        self.send_json(status, {"error": message})

    def check_login_limit(self):
        now = time.time()
        address = self.client_address[0]
        with LOGIN_FAILURES_LOCK:
            failures = [timestamp for timestamp in LOGIN_FAILURES.get(address, []) if now - timestamp < 900]
            LOGIN_FAILURES[address] = failures
            return len(failures) < 8

    def record_login_failure(self):
        with LOGIN_FAILURES_LOCK:
            LOGIN_FAILURES.setdefault(self.client_address[0], []).append(time.time())

    def do_GET(self):
        path = urlsplit(self.path).path
        if path.startswith("/api/"):
            self.handle_api_get(path)
            return
        self.serve_static(path)

    def do_HEAD(self):
        path = urlsplit(self.path).path
        if path.startswith("/api/"):
            self.send_json(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "Method not allowed."})
            return
        self.serve_static(path, head_only=True)

    def serve_static(self, request_path, head_only=False):
        relative = unquote(request_path).lstrip("/") or "index.html"
        target = (ROOT / relative).resolve()
        if target.parent != ROOT or target.name not in {"index.html", "app.js", "style.css"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body_length = target.stat().st_size
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(body_length))
        self.end_headers()
        if not head_only:
            with target.open("rb") as content:
                self.copyfile(content, self.wfile)

    def handle_api_get(self, path):
        if path != "/api/session":
            if path == "/api/state":
                with database_connection() as connection:
                    user = self.current_user(connection)
                    if user is None:
                        self.fail(HTTPStatus.UNAUTHORIZED, "Please sign in.")
                        return
                    row = connection.execute(
                        "SELECT data_json, revision FROM learning_data WHERE user_id = ?",
                        (user["id"],),
                    ).fetchone()
                    payload = {
                        "state": json.loads(row["data_json"]) if row else {},
                        "revision": row["revision"] if row else 0,
                    }
                self.send_json(HTTPStatus.OK, payload)
                return
            self.fail(HTTPStatus.NOT_FOUND, "Not found.")
            return
        with database_connection() as connection:
            user = self.current_user(connection)
            payload = {
                "registration_open": connection.execute("SELECT 1 FROM users LIMIT 1").fetchone() is None,
                "username": user["username"] if user else None,
            }
        self.send_json(HTTPStatus.OK, payload)

    def do_POST(self):
        path = urlsplit(self.path).path
        if not path.startswith("/api/"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not self.has_same_origin():
            self.fail(HTTPStatus.FORBIDDEN, "Request origin did not match this site.")
            return
        try:
            payload = self.read_json()
        except OverflowError as error:
            self.fail(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, str(error))
            return
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            self.fail(HTTPStatus.BAD_REQUEST, "A valid JSON object is required.")
            return
        if path == "/api/register":
            self.handle_register(payload)
        elif path == "/api/login":
            self.handle_login(payload)
        elif path == "/api/logout":
            self.handle_logout()
        else:
            self.fail(HTTPStatus.NOT_FOUND, "Not found.")

    def do_PUT(self):
        path = urlsplit(self.path).path
        if path != "/api/state":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not self.has_same_origin():
            self.fail(HTTPStatus.FORBIDDEN, "Request origin did not match this site.")
            return
        try:
            payload = self.read_json()
        except OverflowError as error:
            self.fail(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, str(error))
            return
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
            self.fail(HTTPStatus.BAD_REQUEST, "A valid JSON object is required.")
            return
        try:
            encoded = json.dumps(payload.get("state"), ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            self.fail(HTTPStatus.BAD_REQUEST, "Learning data must be valid JSON.")
            return
        if not isinstance(payload.get("state"), dict) or type(payload.get("revision")) is not int:
            self.fail(HTTPStatus.BAD_REQUEST, "Learning data must be a JSON object.")
            return
        if len(encoded.encode("utf-8")) > MAX_BODY_BYTES:
            self.fail(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Learning data is too large.")
            return
        with database_connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            user = self.current_user(connection)
            if user is None:
                self.fail(HTTPStatus.UNAUTHORIZED, "Please sign in again.")
                return
            row = connection.execute(
                "SELECT data_json, revision FROM learning_data WHERE user_id = ?",
                (user["id"],),
            ).fetchone()
            current_revision = row["revision"] if row else 0
            if payload["revision"] != current_revision:
                self.send_json(HTTPStatus.CONFLICT, {
                    "error": "Learning data changed on another device.",
                    "state": json.loads(row["data_json"]) if row else {},
                    "revision": current_revision,
                })
                return
            next_revision = current_revision + 1
            connection.execute(
                "INSERT INTO learning_data (user_id, data_json, revision, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET data_json = excluded.data_json, "
                "revision = excluded.revision, updated_at = excluded.updated_at",
                (user["id"], encoded, next_revision, int(time.time())),
            )
        self.send_json(HTTPStatus.OK, {"saved": True, "revision": next_revision})

    def handle_register(self, payload):
        username = payload.get("username")
        password = payload.get("password")
        setup_key = payload.get("setup_key")
        if not isinstance(username, str) or not USERNAME_PATTERN.fullmatch(username):
            self.fail(HTTPStatus.BAD_REQUEST, "Username must be 3–32 letters, numbers, dots, dashes, or underscores.")
            return
        if not isinstance(password, str) or not 12 <= len(password) <= 128:
            self.fail(HTTPStatus.BAD_REQUEST, "Password must be between 12 and 128 characters.")
            return
        if not isinstance(setup_key, str) or len(setup_key) != len(SETUP_KEY) or not hmac.compare_digest(setup_key, SETUP_KEY):
            self.fail(HTTPStatus.FORBIDDEN, "The one-time setup code is invalid or expired.")
            return
        salt = secrets.token_bytes(16)
        password_hash = derive_password(password, salt)
        try:
            with database_connection() as connection:
                connection.execute("BEGIN IMMEDIATE")
                if connection.execute("SELECT 1 FROM users LIMIT 1").fetchone():
                    self.fail(HTTPStatus.CONFLICT, "This personal server already has an account.")
                    return
                cursor = connection.execute(
                    "INSERT INTO users (username, password_salt, password_hash, created_at) VALUES (?, ?, ?, ?)",
                    (username, salt, password_hash, int(time.time())),
                )
                user_id = cursor.lastrowid
                token = self.set_session(connection, user_id)
                connection.execute("INSERT INTO learning_data (user_id, data_json, revision, updated_at) VALUES (?, '{}', 0, ?)", (user_id, int(time.time())))
        except sqlite3.IntegrityError:
            self.fail(HTTPStatus.CONFLICT, "This personal server already has an account.")
            return
        self.send_json(HTTPStatus.CREATED, {"username": username, "state": {}}, cookie=token)

    def handle_login(self, payload):
        username = payload.get("username")
        password = payload.get("password")
        if (
            not isinstance(username, str)
            or not USERNAME_PATTERN.fullmatch(username)
            or not isinstance(password, str)
            or not 1 <= len(password) <= 128
        ):
            self.fail(HTTPStatus.BAD_REQUEST, "Username and password are required.")
            return
        if not self.check_login_limit():
            self.fail(HTTPStatus.TOO_MANY_REQUESTS, "Too many sign-in attempts. Please wait 15 minutes.")
            return
        with database_connection() as connection:
            user = connection.execute(
                "SELECT id, username, password_salt, password_hash FROM users WHERE username = ?",
                (username,),
            ).fetchone()
            salt = user["password_salt"] if user else b"daylight-login-check"
            actual_hash = derive_password(password, salt)
            valid = user is not None and hmac.compare_digest(actual_hash, user["password_hash"])
            if not valid:
                self.record_login_failure()
                self.fail(HTTPStatus.UNAUTHORIZED, "Username or password is incorrect.")
                return
            token = self.set_session(connection, user["id"])
            row = connection.execute(
                "SELECT data_json, revision FROM learning_data WHERE user_id = ?",
                (user["id"],),
            ).fetchone()
            saved_state = json.loads(row["data_json"]) if row else {}
            revision = row["revision"] if row else 0
        self.send_json(HTTPStatus.OK, {"username": user["username"], "state": saved_state, "revision": revision}, cookie=token)

    def handle_logout(self):
        with database_connection() as connection:
            cookies = http.cookies.SimpleCookie()
            try:
                cookies.load(self.headers.get("Cookie", ""))
            except http.cookies.CookieError:
                cookies = http.cookies.SimpleCookie()
            morsel = cookies.get("daylight_session")
            if morsel and morsel.value:
                connection.execute("DELETE FROM sessions WHERE token_hash = ?", (session_token_hash(morsel.value),))
        self.send_json(HTTPStatus.OK, {"signed_out": True}, clear_cookie=True)


def main():
    os.umask(0o077)
    initialize_database()
    host = os.environ.get("DAYLIGHT_HOST", "127.0.0.1")
    port = int(os.environ.get("DAYLIGHT_PORT", "8765"))
    if host not in ("127.0.0.1", "localhost", "::1") and not SECURE_COOKIE:
        raise SystemExit("Remote access requires HTTPS with DAYLIGHT_SECURE_COOKIE=1.")
    server = ThreadingHTTPServer((host, port), DaylightHandler)
    server.daemon_threads = True
    print(f"Daylight is serving on http://{host}:{port}")
    if registration_open():
        print(f"One-time account setup code: {SETUP_KEY}")
        print("Create your personal account in the site before restarting the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Daylight.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
