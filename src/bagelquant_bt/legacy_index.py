"""Streaming schema-1 index decoding, used only by the explicit offline upgrade."""

from __future__ import annotations

import codecs
import hashlib
import json
import tempfile

from .index_blocks import MAX_BYTES
from .store import _json

CHUNK_BYTES = 64 * 1024


class JSONStream:
    """Decode one bounded scalar/record at a time from SQLite's incremental BLOB."""

    def __init__(self, stream):
        self.stream = stream
        self.decoder = codecs.getincrementaldecoder("utf-8")()
        self.buffer = ""
        self.eof = False
        self.peak_buffer = 0

    def fill(self):
        if self.eof:
            return False
        data = self.stream.read(CHUNK_BYTES)
        self.eof = not data
        self.buffer += self.decoder.decode(data, final=self.eof)
        self.peak_buffer = max(self.peak_buffer, len(self.buffer))
        if len(self.buffer) > MAX_BYTES + CHUNK_BYTES:
            raise ValueError("legacy BT metadata record exceeds the streaming budget")
        return bool(data)

    def peek(self):
        while not self.buffer.strip() and not self.eof:
            self.fill()
        self.buffer = self.buffer.lstrip()
        return self.buffer[:1]

    def expect(self, character):
        if self.peek() != character:
            raise ValueError(f"invalid legacy BT JSON: expected {character!r}")
        self.buffer = self.buffer[1:]

    def value(self):
        self.peek()
        while True:
            try:
                value, end = json.JSONDecoder().raw_decode(self.buffer)
                # A number at the current buffer edge may be incomplete.
                if end == len(self.buffer) and not self.eof:
                    self.fill()
                    continue
                self.buffer = self.buffer[end:]
                return value
            except json.JSONDecodeError:
                if not self.fill():
                    raise ValueError(
                        "invalid or truncated legacy BT metadata"
                    ) from None

    def fields(self):
        self.expect("{")
        seen = set()
        if self.peek() != "}":
            while True:
                name = self.value()
                if not isinstance(name, str) or name in seen:
                    raise ValueError("legacy BT metadata keys must be unique")
                seen.add(name)
                self.expect(":")
                yield name
                if self.peek() == "}":
                    break
                self.expect(",")
        self.expect("}")

    def array(self):
        self.expect("[")
        if self.peek() != "]":
            while True:
                yield self.value()
                if self.peek() == "]":
                    break
                self.expect(",")
        self.expect("]")


def read_legacy_index(connection, rowid, reference, *, coordinates=None, groups=None):
    """Verify the legacy canonical digest with finite memory, regardless of key order.

    Canonical array fragments spool to an ephemeral file, then feed the digest in
    canonical field order. A callback consumes each array completely and returns
    its compact replacement. No complete SQLite index cell is materialized.
    """
    arrays, ranges, header, original = {}, {}, {}, {}
    with (
        tempfile.TemporaryFile("w+b") as fragments,
        connection.blobopen(
            "bt_table_index", "schema_json", rowid, readonly=True
        ) as blob,
    ):
        stream = JSONStream(blob)

        def array(name, callback):
            start = fragments.tell()
            checksum = hashlib.sha256(b"[")
            fragments.write(b"[")
            count, columns = 0, None

            def records():
                nonlocal count, columns
                for record in stream.array():
                    encoded = (b"," if count else b"") + _json(record).encode()
                    checksum.update(encoded)
                    fragments.write(encoded)
                    if name == "coordinates":
                        if not isinstance(record, dict) or "_rows" not in record:
                            raise ValueError("invalid legacy BT coordinate record")
                        names = sorted(record)
                        if columns is not None and columns != names:
                            raise ValueError("legacy BT coordinate columns differ")
                        columns = names
                    count += 1
                    yield record
                checksum.update(b"]")
                fragments.write(b"]")
                ranges[name] = (start, fragments.tell() - start)
                arrays[name] = {
                    "digest": checksum.hexdigest(),
                    "rows": count,
                    "columns": columns,
                }

            values = records()
            if callback is None:
                for _ in values:
                    pass
                return {"legacy_array": True}
            result = callback(values)
            if name not in arrays:
                raise ValueError("legacy BT array callback did not consume its input")
            return result

        expected = None
        for name in stream.fields():
            if name not in {"columns", "coordinates", "digest", "evidence", "schema"}:
                raise ValueError(f"unsupported legacy BT index field: {name}")
            if name == "digest":
                expected = stream.value()
                continue
            if name == "coordinates" and stream.peek() == "[":
                header[name] = array(name, coordinates)
                original[name] = None  # Replaced by its spooled canonical bytes.
            elif name == "evidence" and stream.peek() == "{":
                evidence, old_evidence = {}, {}
                for key in stream.fields():
                    if key not in {"axes", "factors", "groups", "schema"}:
                        raise ValueError(f"unsupported legacy BT evidence field: {key}")
                    if key == "groups":
                        evidence[key] = array(key, groups)
                        old_evidence[key] = None
                    else:
                        evidence[key] = old_evidence[key] = stream.value()
                header[name], original[name] = evidence, old_evidence
            else:
                header[name] = original[name] = stream.value()
        original.update(reference=reference, version="bt.table.index.v2")
        digest = hashlib.sha256()

        def emit_object(value, *, evidence=False):
            digest.update(b"{")
            for index, key in enumerate(sorted(value)):
                digest.update((b"," if index else b"") + _json(key).encode() + b":")
                if key == "evidence" and isinstance(value[key], dict):
                    emit_object(value[key], evidence=True)
                elif key in ranges and (
                    key == "coordinates" or (evidence and key == "groups")
                ):
                    offset, remaining = ranges[key]
                    fragments.seek(offset)
                    while remaining:
                        content = fragments.read(min(CHUNK_BYTES, remaining))
                        digest.update(content)
                        remaining -= len(content)
                else:
                    digest.update(_json(value[key]).encode())
            digest.update(b"}")

        emit_object(original)
        if stream.peek() or "schema" not in header or digest.hexdigest() != expected:
            raise ValueError("legacy BT index digest differs; no conversion published")
        return header, arrays, stream.peak_buffer
