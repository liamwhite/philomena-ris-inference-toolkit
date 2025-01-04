from pathlib import Path
import cli
import csv
import db
import fire
import requests
import threading


def load(file: str, database: str, delimiter: str = ",", quotechar: str = '"'):
    """
    Prepare all given rows from the given CSV file for download, but do not start downloading them.

    CSV format should have rows in the form of `id,url,path`.

    :param file: path to the CSV file
    :param database: path to the database
    :param delimiter: delimiter for CSV data
    :param quotechar: quote character for CSV data
    """
    conn = db.connect(database_path=database)

    with open(file, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter, quotechar=quotechar)
        db.load_pending_downloads(conn, reader)


def local(file: str, database: str, delimiter: str = ",", quotechar: str = '"'):
    """
    Assume all given rows from the given CSV file exist and are ready to process.

    CSV format should have rows in the form of `id,path`.

    :param file: path to the CSV file
    :param database: path to the database
    :param delimiter: delimiter for CSV data
    :param quotechar: quote character for CSV data
    """
    conn = db.connect(database_path=database)

    with open(file, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter, quotechar=quotechar)
        db.load_images(conn, reader)


def _download_and_write_to_disk(session: requests.Session, url: str, path: str):
    response = session.get(url)
    response.raise_for_status()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(response.content)


def _process_all_records(database: str, lock: threading.Lock):
    conn = db.connect(database_path=database)
    session = requests.Session()

    for id, url, path in db.get_images_for_download(conn, lock):
        try:
            _download_and_write_to_disk(session, url, path)
            db.pass_image_download(conn, id, path, lock)
            print(f"\r{path}    ", end="")
        except requests.RequestException:
            db.fail_image_download(conn, id, lock)
            print(f"\nFailed to fetch {id}")


def download(concurrency: int, database: str, base_path: str | None = None):
    """
    Download all images that are not yet fetched.
    :param concurrency: number of concurrent HTTP requests allowed
    :param database: path to the database
    """
    lock = threading.Lock()
    threads = []

    cli.force_interrupt_shutdown()
    cli.set_base_path(base_path)
    db.reset_in_progress_downloads(database_path=database)

    for _ in range(0, concurrency):
        thread = threading.Thread(target=_process_all_records, args=(database, lock))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    print("\nDone")


def reset_failed_downloads(database: str):
    """
    Clear failure state of all pending downloads.
    :param database: path to the database
    """
    db.reset_failed_and_in_progress_downloads(database_path=database)


if __name__ == "__main__":
    fire.Fire()
