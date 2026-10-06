import http.cookiejar
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import HTTPCookieProcessor, Request, build_opener

import server


class QuietDaylightHandler(server.DaylightHandler):
    def log_message(self, format_string, *args):
        pass


class AccountServerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_database = server.DATABASE
        self.original_setup_key = server.SETUP_KEY
        server.DATABASE = Path(self.temp_dir.name) / "daylight.sqlite3"
        server.SETUP_KEY = "one-time-test-code"
        server.LOGIN_FAILURES.clear()
        server.initialize_database()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), QuietDaylightHandler)
        self.base_url = f"http://127.0.0.1:{self.httpd.server_port}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.opener = self.new_opener()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join()
        server.DATABASE = self.original_database
        server.SETUP_KEY = self.original_setup_key
        self.temp_dir.cleanup()

    def new_opener(self):
        jar = http.cookiejar.CookieJar()
        opener = build_opener(HTTPCookieProcessor(jar))
        opener.test_cookie_jar = jar
        return opener

    def request(self, opener, path, method="GET", payload=None, origin=None):
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if origin is not None:
            headers["Origin"] = origin
        elif method in ("POST", "PUT"):
            headers["Origin"] = self.base_url
        request = Request(f"{self.base_url}{path}", data=data, headers=headers, method=method)
        try:
            response = opener.open(request)
        except HTTPError as error:
            response = error
        body = response.read()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = body
        return response.code, payload

    def test_private_account_syncs_and_checks_revisions(self):
        status, session = self.request(self.opener, "/api/session")
        self.assertEqual(status, 200)
        self.assertTrue(session["registration_open"])

        status, registered = self.request(self.opener, "/api/register", "POST", {
            "username": "learner",
            "password": "a-secure-personal-password",
            "setup_key": server.SETUP_KEY,
        })
        self.assertEqual(status, 201)
        self.assertEqual(registered["username"], "learner")
        self.assertTrue(any(cookie.has_nonstandard_attr("HttpOnly") for cookie in self.opener.test_cookie_jar))

        status, _ = self.request(self.opener, "/api/register", "POST", {
            "username": "another",
            "password": "another-secure-password",
            "setup_key": server.SETUP_KEY,
        })
        self.assertEqual(status, 409)

        local = {"words": [{"word": "hello"}], "completedLessons": ["w1d1"]}
        status, result = self.request(self.opener, "/api/state", "PUT", {"state": local, "revision": 0})
        self.assertEqual(status, 200)
        self.assertEqual(result["revision"], 1)

        second_device = self.new_opener()
        status, login = self.request(second_device, "/api/login", "POST", {
            "username": "learner",
            "password": "a-secure-personal-password",
        })
        self.assertEqual(status, 200)
        self.assertEqual(login["state"], local)
        self.assertEqual(login["revision"], 1)

        status, conflict = self.request(self.opener, "/api/state", "PUT", {"state": {}, "revision": 0})
        self.assertEqual(status, 409)
        self.assertEqual(conflict["revision"], 1)
        self.assertEqual(conflict["state"], local)

        status, _ = self.request(self.opener, "/api/logout", "POST", {})
        self.assertEqual(status, 200)
        status, _ = self.request(second_device, "/api/state")
        self.assertEqual(status, 200)

    def test_rejects_cross_origin_changes_and_private_files(self):
        status, _ = self.request(self.opener, "/api/login", "POST", {}, origin="https://attacker.invalid")
        self.assertEqual(status, 403)
        status, _ = self.request(self.opener, "/daylight.sqlite3")
        self.assertEqual(status, 404)
        status, _ = self.request(self.opener, "/server.py")
        self.assertEqual(status, 404)

    def test_registration_requires_setup_code(self):
        status, _ = self.request(self.opener, "/api/register", "POST", {
            "username": "learner",
            "password": "a-secure-personal-password",
            "setup_key": "wrong",
        })
        self.assertEqual(status, 403)
        status, _ = self.request(self.opener, "/api/register", "POST", {
            "username": "learner",
            "password": "short",
            "setup_key": server.SETUP_KEY,
        })
        self.assertEqual(status, 400)
        status, session = self.request(self.opener, "/api/session")
        self.assertTrue(session["registration_open"])


if __name__ == "__main__":
    unittest.main()
