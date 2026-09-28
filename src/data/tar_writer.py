import json

import numpy as np
import webdataset as wds

from .utils import (
    get_nonexistant_path,
)


def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError("Unkown type:", type(obj))


class WebDatasetWriter:
    def __init__(self, config) -> None:
        self.cfg = config
        self.sink = None

    def _is_jsonable(self, x):
        try:
            json.dumps(x, default=default)
            return True
        except (TypeError, OverflowError):
            return False

    def _get_serializable_data(self, data):
        keys_to_delete = []
        for key, value in data.items():
            if not self._is_jsonable(value):
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del data[key]
        return data

    def create_tar_file(self, file_name, write_path):
        # Check if file already exists, increment if so
        if self.cfg["data_writer"]["shard_write"]:
            path_to_file = write_path + file_name + "_%06d.tar.gz"
        else:
            path_to_file = write_path + file_name + ".tar.gz"

        # Create a folder
        write_path = get_nonexistant_path(path_to_file)
        write_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a tar file
        if self.cfg["data_writer"]["shard_write"]:
            max_count = self.cfg["data_writer"]["shard_maxcount"]
            self.sink = wds.ShardWriter(
                str(write_path), maxcount=max_count, compress=True
            )
        else:
            self.sink = wds.TarWriter(str(write_path), compress=True)

    def write(self, data):
        if self.sink is None:
            raise FileNotFoundError(
                "Please call create_tar_file() method before calling the write method"
            )
        self.sink.write(data)

    def close(self):
        self.sink.close()
