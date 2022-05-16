#!/usr/bin/env python

import sys
import subprocess
from io import RawIOBase


class StreamFilter(RawIOBase):
    def __init__(self, conds, stream):
        super().__init__()
        self.conds = conds
        self.stream = stream

    def readinto(self, _):
        pass

    def write(self, msg):
        if all(cond(msg) for cond in self.conds):
            self.stream.write(msg)
        else:
            pass


class CleanLog(object):
    def __init__(self, filter_, stream_name):
        self.filter = filter_
        self.stream_name = stream_name
        self.old_stream = getattr(sys, stream_name)

    def __enter__(self):
        setattr(sys, self.stream_name, self.filter)

    def __exit__(self, exc_type, exc_value, traceback):
        setattr(sys, self.stream_name, self.old_stream)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise TypeError("请指定需要运行的脚本！")

    tar_file = sys.argv[1]
    gdal_filter = StreamFilter([
        lambda msg: "Sum of Photometric type-related color channels and ExtraSamples doesn't match SamplesPerPixel." not in msg
    ], sys.stdout)
    with CleanLog(gdal_filter, 'stdout'):
        proc = subprocess.Popen(
            ["python", tar_file],
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE,
            text=True)
        while True:
            try:
                out_line = proc.stdout.readline()
                if out_line == '' and proc.poll() is not None:
                    break
                if out_line:
                    print(out_line, end='')
            except KeyboardInterrupt:
                import signal
                proc.send_signal(signal.SIGINT)
