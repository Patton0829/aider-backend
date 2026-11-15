#!/usr/bin/env python3
"""One-time local MySQL provisioning. Passwords never enter command arguments or logs."""
import argparse
import os
from pathlib import Path
import re
import secrets
import subprocess
from tempfile import NamedTemporaryFile

from sqlalchemy.engine import URL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admin-user", default="root")
    parser.add_argument("--port", type=int, default=3306)
    parser.add_argument("--mysql-bin", default="mysql")
    parser.add_argument("--database", default="aider_rag")
    parser.add_argument("--user", default="aider_rag")
    args = parser.parse_args()
    if not all(re.fullmatch(r"[A-Za-z0-9_]{1,32}", value) for value in [args.database, args.user]):
        parser.error("Database/user names must contain 1-32 ASCII letters, digits or underscores")
    if not 1 <= args.port <= 65535:
        parser.error("Invalid MySQL port")
    root = Path(__file__).resolve().parents[1]
    environment = root / ".env"
    if not environment.is_file():
        parser.error("Run scripts/setup.sh first to create the local .env")
    # Include all common password-policy character classes without SQL quote characters.
    password = "Rag9!" + secrets.token_hex(24) + "aA1_"
    url = URL.create("mysql+asyncmy", username=args.user, password=password, host="127.0.0.1", port=args.port, database=args.database)
    updates = {"KNOWLEDGE_HISTORY_STORE": "mysql", "KNOWLEDGE_DATABASE_URL": url.render_as_string(hide_password=False)}
    lines = []
    for line in environment.read_text().splitlines():
        key = line.split("=", 1)[0].strip()
        if key in updates:
            lines.append(key + "=" + updates[key])
        else:
            lines.append(line)
    present = {line.split("=", 1)[0].strip() for line in lines}
    lines.extend(key + "=" + value for key, value in updates.items() if key not in present)
    # CREATE USER intentionally has no IF NOT EXISTS/ALTER USER: never rotate an existing account.
    sql = (
        f"CREATE DATABASE IF NOT EXISTS `{args.database}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;\n"
        f"CREATE USER '{args.user}'@'127.0.0.1' IDENTIFIED BY '{password}';\n"
        f"GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, INDEX, ALTER, REFERENCES ON `{args.database}`.* TO '{args.user}'@'127.0.0.1';\n"
    )
    temp_path = None
    try:
        private_dir = root / "storage" / "rag"
        private_dir.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=private_dir, prefix=".history-env-", delete=False) as temporary:
            temp_path = Path(temporary.name)
            os.chmod(temp_path, 0o600)
            temporary.write("\n".join(lines) + "\n")
        print("Enter the existing MySQL administrator password when prompted. Existing accounts are not altered.", flush=True)
        subprocess.run([args.mysql_bin, "--host=localhost", f"--port={args.port}", f"--user={args.admin_user}", "--password"], input=sql, text=True, check=True)
        os.replace(temp_path, environment)
        temp_path = None
        print("Dedicated history database/user created and .env updated. Run .venv/bin/python -m app.manage db-init.")
        return 0
    except (OSError, subprocess.CalledProcessError):
        print("Provisioning did not complete; .env was not replaced. Check administrator access or use an existing dedicated database account.")
        return 1
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
