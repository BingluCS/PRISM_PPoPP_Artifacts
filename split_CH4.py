import numpy as np
import os
import sys

def extract_ch4(input_file, ch4_index=3):
    dtype = np.float64
    ch4_size = 500 * 500 * 500  
    bytes_per_elem = np.dtype(dtype).itemsize
    read_bytes = ch4_size * bytes_per_elem

    folder = os.path.dirname(input_file)
    outfile = os.path.join(folder, "CH4.d64")

    with open(input_file, "rb") as f:
        raw = f.read(read_bytes)

    ch4 = np.frombuffer(raw, dtype=dtype).reshape(500, 500, 500)

    ch4.tofile(outfile)
    print(f"save CH4 to:{outfile}")
    return outfile

input_path = sys.argv[1]
extract_ch4(input_path)
