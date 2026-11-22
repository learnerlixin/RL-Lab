"""Small CSV metrics logger with an explicit, stable schema."""
import csv
from pathlib import Path
from typing import Iterable, Mapping


class CSVLogger:
    def __init__(self, path, fieldnames: Iterable[str]):
        self.path = Path(path)
        self.fieldnames = list(fieldnames)
        if not self.fieldnames or len(set(self.fieldnames)) != len(self.fieldnames):
            raise ValueError("fieldnames must be non-empty and unique")
        if any(not isinstance(name, str) or not name for name in self.fieldnames):
            raise ValueError("fieldnames must be non-empty strings")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        existing = self.path.exists() and self.path.stat().st_size > 0
        if existing:
            with self.path.open("r", newline="", encoding="utf-8") as stream:
                header = next(csv.reader(stream), None)
            if header != self.fieldnames:
                raise ValueError(
                    f"existing CSV schema {header!r} does not match {self.fieldnames!r}"
                )
        self._file = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames,
                                      extrasaction="raise")
        if not existing:
            self._writer.writeheader()
            self._file.flush()

    def log(self, metrics: Mapping):
        if self._file.closed:
            raise RuntimeError("cannot log after CSVLogger has been closed")
        missing = set(self.fieldnames) - set(metrics)
        extra = set(metrics) - set(self.fieldnames)
        if missing or extra:
            raise ValueError(f"metric schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
        self._writer.writerow(dict(metrics))
        self._file.flush()

    def close(self):
        if not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


__all__ = ["CSVLogger"]
