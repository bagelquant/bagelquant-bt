"""Finite-memory combination of exact coordinate evidence.

The disposable sorter is scratch space, never a persisted proof authority.
Returned daily digests have the original canonical JSON ordering and identity.
"""

import hashlib
import json
import sqlite3
import tempfile
from contextlib import closing, contextmanager
from pathlib import Path


@contextmanager
def evidence_groups():
    with tempfile.TemporaryDirectory(prefix="bt-evidence-sort-") as directory:
        with closing(sqlite3.connect(Path(directory) / "groups.sqlite")) as connection:
            connection.execute("PRAGMA cache_size=-4096")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("PRAGMA journal_mode=OFF")
            connection.create_function(
                "sum64",
                2,
                lambda a, b: format((int(a, 16) + int(b, 16)) % (1 << 64), "016x"),
            )
            connection.execute(
                "CREATE TABLE groups(key TEXT PRIMARY KEY,day TEXT,n INTEGER,a TEXT,b "
                "TEXT)"
            )
            yield connection


def add_groups(connection, records):
    connection.executemany(
        "INSERT INTO groups VALUES(?,?,?,?,?) ON CONFLICT(key) DO UPDATE SET "
        "n=n+excluded.n,a=sum64(a,excluded.a),b=sum64(b,excluded.b)",
        records,
    )


def date_identities(connection, axes):
    from .store import _json

    identities = {}
    current_day, digest = None, None
    for key, day, n, a, b in connection.execute(
        "SELECT * FROM groups ORDER BY day,key"
    ):
        if digest is None or day != current_day:
            if digest is not None:
                digest.update(b"]}")
                identities[current_day] = digest.hexdigest()
            current_day = day
            digest = hashlib.sha256(b'{"axes":' + _json(axes).encode() + b',"groups":[')
            separator = b""
        digest.update(separator + _json([*json.loads(key), n, a, b]).encode())
        separator = b","
    if digest is not None:
        digest.update(b"]}")
        identities[current_day] = digest.hexdigest()
    return identities
