from typing import Iterable, Tuple
import struct
import sqlite3
import threading

DOWNLOAD_READY = 0
DOWNLOAD_IN_PROGRESS = 1
DOWNLOAD_FAILED = 2

PROCESS_READY = 0
PROCESS_IN_PROGRESS = 1
PROCESS_CORRUPT = 2
PROCESS_SUCCEEDED = 3


def _enable_wal_mode(conn: sqlite3.Connection):
    conn.execute("pragma journal_mode=wal")


def _setup_tables(conn: sqlite3.Connection):
    # Images which have yet to be downloaded
    conn.execute(
        "create table if not exists pending_downloads (id integer primary key not null, url text not null, path text not null, download_state integer default 0) strict"
    )
    conn.execute(
        "create index if not exists index_pending_downloads_on_download_state_and_id on pending_downloads (download_state, id)"
    )
    # Images which are assumed to exist
    conn.execute(
        "create table if not exists images (id integer not null primary key, path text not null, process_state integer not null default 0) strict"
    )
    conn.execute(
        "create index if not exists index_images_on_process_state_and_id on images (process_state, id)"
    )
    # Output feature vectors for images
    conn.execute(
        "create table if not exists image_vectors (id integer primary key not null, vector blob not null) strict"
    )


def connect(database_path: str) -> sqlite3.Connection:
    """
    Returns a new database connection configured in WAL mode.

    :param database_path: The local path to the database.
    """
    conn = sqlite3.connect(database_path)
    _enable_wal_mode(conn)
    _setup_tables(conn)
    return conn


def load_pending_downloads(
    conn: sqlite3.Connection, rows: Iterable[Tuple[int, str, str]]
):
    """
    Import all given rows from the iterator into the pending_downloads table.
    :param rows: Iterable object returning (id, url, path)
    """
    with conn:
        for row in rows:
            conn.execute(
                "insert into pending_downloads (id, url, path) values (?, ?, ?)", row
            )


def load_images(conn: sqlite3.Connection, rows: Iterable[Tuple[int, str, str]]):
    """
    Import all given rows from the iterator into the images table.
    :param rows: Iterable object returning (id, path)
    """
    with conn:
        for row in rows:
            conn.execute("insert into images (id, path) values (?, ?, ?)", row)


def reset_in_progress_downloads(database_path: str):
    """
    Mark all pending downloads as not in progress during a fresh run.
    """
    conn = connect(database_path)
    with conn:
        conn.execute(
            "update pending_downloads set download_state = ? where download_state = ?",
            (DOWNLOAD_READY, DOWNLOAD_IN_PROGRESS),
        )


def reset_failed_and_in_progress_downloads(database_path: str):
    """
    Reset failed state from pending downloads.
    """
    conn = connect(database_path)
    with conn:
        conn.execute(
            "update pending_downloads set download_state = ?", (DOWNLOAD_READY,)
        )


def get_images_for_download(
    conn: sqlite3.Connection, lock: threading.Lock
) -> Iterable[Tuple[int, str, str]]:
    """
    Gets data for image IDs to download and marks them as in progress.
    Yields tuples of (id, url, path).
    """
    while True:
        with lock, conn:
            row = conn.execute(
                "select id, url, path from pending_downloads where download_state = ? order by id asc limit 1",
                (DOWNLOAD_READY,),
            ).fetchone()

            if not row:
                return

            id, url, path = row
            conn.execute(
                "update pending_downloads set download_state = ? where id = ?",
                (DOWNLOAD_IN_PROGRESS, id),
            )

        yield (id, url, path)


def pass_image_download(
    conn: sqlite3.Connection, id: int, path: str, lock: threading.Lock
):
    """
    Marks the given image as successfully downloaded and stores its path.
    """
    with lock, conn:
        conn.execute("insert into images (id, path) values (?, ?)", (id, path))
        conn.execute("delete from pending_downloads where id = ?", (id,))


def fail_image_download(conn: sqlite3.Connection, id: int, lock: threading.Lock):
    """
    Marks the given image as not successfully downloaded.
    """
    with lock, conn:
        conn.execute(
            "update pending_downloads set download_state = ? where id = ?",
            (DOWNLOAD_FAILED, id),
        )


def reset_in_progress_process(database_path: str):
    """
    Mark all pending images as not in being processed during a fresh run.
    """
    conn = connect(database_path)
    with conn:
        conn.execute(
            "update images set process_state = ? where process_state = ?",
            (PROCESS_READY, PROCESS_IN_PROGRESS),
        )


def get_images_for_process(
    conn: sqlite3.Connection, lock: threading.Lock
) -> Iterable[Tuple[int, str]]:
    """
    Gets data for image IDs to process and marks them as in progress.
    Yields tuples of (id, path).
    """
    while True:
        with lock, conn:
            row = conn.execute(
                "select id, path from images where process_state = ? order by id asc limit 1",
                (PROCESS_READY,),
            ).fetchone()

            if not row:
                return

            id, path = row
            conn.execute(
                "update images set process_state = ? where id = ?",
                (PROCESS_IN_PROGRESS, id),
            )

        yield (id, path)


def pass_image_process(
    conn: sqlite3.Connection, id: int, features: list[float], lock: threading.Lock
):
    """
    Marks the given image as successfully processed and stores its features.
    """
    vector = struct.pack("<768f", *features)
    with lock, conn:
        conn.execute(
            "insert into image_vectors (id, vector) values (?, ?)", (id, vector)
        )
        conn.execute(
            "update images set process_state = ? where id = ?", (PROCESS_SUCCEEDED, id)
        )


def fail_image_process(conn: sqlite3.Connection, id: int, lock: threading.Lock):
    """
    Marks the given image as not successfully processed.
    """
    with lock, conn:
        conn.execute(
            "update images set process_state = ? where id = ?", (PROCESS_CORRUPT, id)
        )


def _fetch_batched(cursor: sqlite3.Cursor, size: int) -> Iterable[tuple]:
    while True:
        rows = cursor.fetchmany(size)

        if not rows:
            break

        yield from rows


def get_image_features(conn: sqlite3.Connection) -> Iterable[Tuple[int, list[float]]]:
    """
    Gets data for all available image features.
    Generates tuples of (id, features).
    """
    cursor = conn.execute("select id, vector from image_vectors order by id")
    for id, vector in _fetch_batched(cursor, 1000):
        yield (id, vector)
