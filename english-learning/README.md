# Daylight English

Daylight is a personal English-learning site with a wordbook, flashcards, a daily checklist, writing practice, and saved progress for each study route. It includes a 12-week beginner plan and original study guides for New Concept English Books 1–3, Cambridge grammar at beginner/intermediate/advanced levels, and IELTS preparation alongside IELTS 真经. The study guides are original topic plans; use them with your own books and edition.

## Run locally

```sh
cd english-learning
python3 server.py
```

Open `http://127.0.0.1:8765`. Python's standard library is the only dependency. The first server start prints a one-time setup code in the terminal. Select **首次创建账号**, enter the code, choose a username, and set a password of at least 12 characters. If you restart before creating the account, use the new code printed by the server. This personal server permits only one account; account creation closes after the first successful registration.

The site continues to keep a local copy in the browser. On first sign-in, local words, completed lessons, activity dates, checklist progress, and writing are merged into the account; duplicate words are combined and the longer writing is kept. While signed in, changes sync automatically. A revision check detects updates made on another device and merges them before retrying. Sign out restores the guest data saved on that browser.

## Access from other devices

For multi-device access, host this folder on a private server and put an HTTPS reverse proxy such as Caddy or nginx in front of the app. Run the app bound to localhost behind that proxy:

```sh
DAYLIGHT_HOST=127.0.0.1 DAYLIGHT_PORT=8765 DAYLIGHT_SECURE_COOKIE=1 python3 server.py
```

Forward HTTPS traffic to `127.0.0.1:8765`, preserve the `Host` header, and do not expose port 8765 directly to the internet. Set a strong server and account password, keep the one-time setup code private, and back up `daylight.sqlite3` while the server is stopped. The database contains account credentials (as salted password hashes) and private learning data. There is no email-based password recovery; keep a protected database backup.

The server uses only same-origin requests; it does not contact third-party services. Without the Python server, opening `index.html` directly still provides the browser-local learning tools, but account login and sync are unavailable.
