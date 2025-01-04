import db
import fire
import struct
import ujson as json


def lines(database: str, output: str):
    """
    Export all image vectors as NDJSON.
    :param database: path to the database
    :param output: path to the output NDJSON file
    """
    conn = db.connect(database_path=database)
    unpack_features = struct.Struct("<768f").unpack

    with open(output, "w") as file:
        for id, vector in db.get_image_features(conn):
            file.write(json.dumps({"id": id, "features": unpack_features(vector)}))
            file.write("\n")


def binary(database: str, output: str):
    """
    Export all image vectors as binary.
    :param database: path to the database
    :param output: path to the output binary file
    """
    conn = db.connect(database_path=database)
    pack_id = struct.Struct("<I").pack

    with open(output, "wb") as file:
        for id, vector in db.get_image_features(conn):
            file.write(pack_id(id))
            file.write(vector)


if __name__ == "__main__":
    fire.Fire()
