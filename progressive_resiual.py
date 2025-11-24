import subprocess
import sys
from matplotlib import text
import numpy as np
import shutil
import os
import re
import csv

if len(sys.argv) != 4:
    print("Usage: python progressive_retrieval.py <dataset_name> <input_file>")
    sys.exit(1)

csv_file = sys.argv[3]

header = ["type", "field", 'error',"comp_th", "decomp_th", "ratio", "psnr"]
bit_rate = [4,3,4,3,5,2,4,3,4]
acc_value = [0, 0, 0]
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_row(row):
    with open(csv_file, "a", newline="") as f:  # "a" = append
        writer = csv.writer(f)
        writer.writerow(row)

def compute_psnr(input_file, decompressed_file, ddtype, shape):
    num_elements = np.prod(shape)
    original = np.fromfile(input_file, dtype=ddtype, count=num_elements).reshape(shape)
    reconstructed = np.fromfile(decompressed_file, dtype=ddtype, count=num_elements).reshape(shape)

    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float("inf")
    data_range = original.max() - original.min()
    psnr = 20 * np.log10(data_range) - 10 * np.log10(mse)
    diff = np.abs(original - reconstructed)
    max_diff = diff.max() 
    # print(f"  Max_E = {max_diff}\n  Max_RE = {max_diff / data_range}\n  PSNR = {psnr}")
    return psnr, max_diff / data_range, max_diff

def run_prism(shape, data_type, input_file, e, nums, errors_list):
    data_type_para = '-f'
    if(data_type == "float"):
        data_type_para = "-f"
    else:
        data_type_para = "-d"
    compressed_file = input_file + ".prism"
    decompressed_file = input_file + ".prism.out"
    cmd = [
        "../PRISM/build/prism",
        "-3", str(shape[0]), str(shape[1]), str(shape[2]),
        data_type_para,
        "-i", input_file,
        "-z", compressed_file,
        "-x", decompressed_file,
        "-R", str(e),
        "--report", "time,cr",
        "-prog", "-errors", nums,
    ] + [str(x) for x in errors_list]
    # subprocess.run(cmd, check=True)
    # result = 0
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)

    return decompressed_file, result

def run_cusz(shape, data_type, input_file, e, mode = 'abs'):
    shape_str = 'x'.join(map(str, shape))
    data_type_para = 'f32'
    if(data_type == "float"):
        data_type_para = "f32"
    else:
        data_type_para = "f64"
    compressed_file = input_file + ".cusza"
    decompressed_file = input_file + ".cuszx"
    cmd_com = [
        "cuSZ/build/cusz",
        "-l", shape_str,
        "-t", data_type_para,
        "-i", input_file,
        "-z",
        "-m", mode, "-e", "{:.8f}".format(e) ,
        "--report", "cr"
    ]
    cmd_decom = [
        "cuSZ/build/cusz",
        "-i", compressed_file,
        "-x",
    ]
    result = subprocess.run(cmd_com, check=True, capture_output=True, text=True)
    comth_match = re.search(r"TOTAL\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)", result.stdout)
    cr_match = re.search(r"data::comp_metric::CR\s+([0-9.eE+-]+)", result.stdout)
    if comth_match:
        com_th = float(comth_match.group(2))
    cr_value = float(cr_match.group(1)) if cr_match else None
    result = subprocess.run(cmd_decom, check=True, capture_output=True, text=True)
    decomth_match = re.search(r"TOTAL\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)", result.stdout)
    if decomth_match:
        decomp_th = float(decomth_match.group(2))
    return decompressed_file, np.array([com_th, decomp_th, cr_value])

def run_cuszp(shape, data_type, input_file, e, mode = 'abs'):
    data_type_para = 'f32'
    if(data_type == "float"):
        data_type_para = "f32"
    else:
        data_type_para = "f64"
    compressed_file = input_file + ".cuszp.cmp"
    decompressed_file = input_file + ".cuszp.dec"
    cmd = [
        "cuSZp/build/examples/bin/cuSZp",
        "-t", data_type_para,
        "-i", input_file,
        "-x", compressed_file,
        "-o", decompressed_file,
        "-m", "outlier",
        "-eb", mode, str(e),
    ]
    result= subprocess.run(cmd, check=True, capture_output=True, text=True)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", result.stdout)
    numbers = [float(x) for x in numbers[:3]]
    return decompressed_file, np.array(numbers)

def run_zfp(shape, data_type, input_file, bit):
    data_type_para = 'f32'
    if(data_type == "float"):
        data_type_para = "-f"
    else:
        data_type_para = "-d"
    compressed_file = input_file + ".cuszp.cmp"
    decompressed_file = input_file + ".cuszp.dec"
    cmd = [
        "cuZFP/build/bin/zfp",
        data_type_para,
        "-3", str(shape[0]), str(shape[1]), str(shape[2]),
        "-i", input_file,
        "-s", 
        "-z", compressed_file,
        "-o", decompressed_file,
        "-x", "cuda",
        "-r", str(bit),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    # print(result.stdout)
    match_encode = re.search(r"encode3 rate:\s*([\d.]+)", result.stdout)
    encode3_rate = float(match_encode.group(1)) if match_encode else None

    match_decode = re.search(r"decode3 rate:\s*([\d.]+)", result.stdout)
    decode3_rate = float(match_decode.group(1)) if match_decode else None

    match_ratio = re.search(r"ratio=([\d.]+)", result.stderr)
    ratio = float(match_ratio.group(1)) if match_ratio else None
    numbers = [encode3_rate, decode3_rate, ratio]

    return decompressed_file, np.array(numbers)

def run_hpmdr(shape, data_type, input_file, nums, errors_list):
    # shape_str = 'x'.join(map(str, shape))
    data_type_para = 'f32'
    if(data_type == "float"):
        data_type_para = "s"
    else:
        data_type_para = "d"
    compressed_file = input_file + ".mgard"
    decompressed_file = input_file + ".mgard.dec"
    cmd_comp = [
        "HP-MDR/build-cuda-hopper/mgard/bin/mdr-x",
        "--refactor",
        "--input", input_file,
        "--output", compressed_file,
        "-dt", data_type_para,
        "-dim", "3", str(shape[2]), str(shape[1]), str(shape[0]),
        "-dd", "max-dim",
        "-d", "cuda",
        "-v", "3",
    ]
    cmd_decom = [
        "HP-MDR/build-cuda-hopper/mgard/bin/mdr-x",
        "--reconstruct",
        "--input", compressed_file,
        "-o", decompressed_file,
        "-g", input_file,
        "-dt", data_type_para,
        "-dim", "3", str(shape[2]), str(shape[1]), str(shape[0]),
        "-s", "inf",
        "-ar", "0",
        "-v", "3",
        "-d", "cuda",
        "-m", "abs", "-me", nums] + [str(x) for x in errors_list]
    
    cmp_result = subprocess.run(cmd_comp, check=True, capture_output=True, text=True)
    decmp_result = subprocess.run(cmd_decom, check=True, capture_output=True, text=True)
    return decompressed_file, cmp_result, decmp_result


def accumulate_residual(accumulate_file, decompressed_file, ddtype):
    B = np.fromfile(decompressed_file, dtype=ddtype)
    if not os.path.exists(accumulate_file):
        shutil.copyfile(decompressed_file, accumulate_file)
        return
    A = np.fromfile(accumulate_file, dtype=ddtype)
    if A.size != B.size and abs(A.size - B.size) != 192:
        raise ValueError(f"A size != B size")
    A += B
    A.tofile(accumulate_file)

def compute_residual(input_file, accumulate_file, residual_file, ddtype):
    B = np.fromfile(accumulate_file, dtype=ddtype)
    A = np.fromfile(input_file, dtype=ddtype)
    # if A.size != B.size and abs(A.size - B.size) != 192:
    #     raise ValueError(f"A size != B size")
    A -= B[:A.size]
    A.tofile(residual_file)

def run_progressive_residual(shape, data_type, input_file, compressor):
    ddtype = np.float64
    bits = 8 * 4
    if data_type == "double":
        ddtype = np.float64
        errors = [1E-1, 1E-2, 1E-3, 1E-4, 1E-5, 1E-6, 1E-7, 1E-8, 1E-9]
        init_errors = "1E-9"
        bits = 64
    else:
        ddtype = np.float32
        errors = [1E-1, 5E-2, 1E-2, 5E-3, 1E-3, 5E-4, 1E-4, 5E-5, 1E-5]
        init_errors = "1E-5"
    value_range = 0
    data = np.memmap(input_file, dtype=ddtype, mode="r")
    value_range = data.max() - data.min()


    accumulate_file = input_file + ".acc"
    residual_file = input_file + ".res"
    if os.path.exists(accumulate_file):
        os.remove(accumulate_file)
    if compressor == "cuSZ":
        decompressed_file, numbers = run_cusz(shape, data_type, input_file,  errors[0] * value_range, 'abs')
    elif compressor == "cuSZp":
        decompressed_file, numbers = run_cuszp(shape, data_type, input_file,  errors[0] * value_range, 'abs')
    elif compressor == "cuZFP":
        decompressed_file, numbers = run_zfp(shape, data_type, input_file, bit_rate[0])
        accumulate_residual(accumulate_file, decompressed_file, ddtype)
        acc_value = numbers
        psnr = compute_psnr(input_file, accumulate_file, ddtype, shape)
        out_row = [compressor, sys.argv[1], psnr[1], acc_value[0], acc_value[1], acc_value[2], psnr[0]]
        append_row(out_row)
        compute_residual(input_file, accumulate_file, residual_file, ddtype)

        for r in bit_rate[1:]:
            decompressed_file, numbers = run_zfp(shape, data_type, residual_file, r)
            accumulate_residual(accumulate_file, decompressed_file, ddtype)
            psnr = compute_psnr(input_file, accumulate_file, ddtype, shape)
            acc_value = 1 / (1 / acc_value + 1 / numbers)
            out_row = [compressor, sys.argv[1], psnr[1], acc_value[0], acc_value[1], acc_value[2], psnr[0]]
            append_row(out_row)

            compute_residual(input_file, accumulate_file, residual_file, ddtype)
        return
    elif compressor == "PRISM":
        decompressed_file, result = run_prism(shape, data_type, input_file, init_errors, '-9', np.array(errors))
        bitrates = re.findall(r"Bit rate:\s*([0-9.+Ee-]+)", result.stdout)
        psnr = re.findall(r"PSNR\s*=\s*([0-9.+Ee-]+)", result.stdout)
        for i, e in enumerate(errors):
            decompressed_file, result = run_prism(shape, data_type, input_file, init_errors, '-1', [e])
            dec_th = float(re.search(r"itotal.*?\d+\.\d+.*?(\d+\.\d+)", result.stdout).group(1))
            comp_th = float(re.search(r"(?<!i)total.*?\d+\.\d+.*?(\d+\.\d+)", result.stdout).group(1))
            out_row = [compressor, sys.argv[1], e, comp_th, dec_th, bits / float(bitrates[i]), psnr[i]]
            append_row(out_row)
        return
    elif compressor == "HP-MDR":
        total_bytes = (shape[0] * shape[1] * shape[2] * bits / 8)
        decompressed_file, _, decmp_result = run_hpmdr(shape, data_type, input_file, '9', np.array(errors) * value_range)
        additional_bytes = re.findall(r'Additional\s+(\d+)\s+bytes', decmp_result.stdout)
        psnr_list = re.findall(r'PSNR:\s*([+-]?\d+(?:\.\d+)?)', decmp_result.stdout)
        for i, e in enumerate(errors):
            decompressed_file, comp_result, decmp_result = \
                run_hpmdr(shape, data_type, input_file, '1', [e * value_range])
            
            match_die = re.search(r"Decompose \+ Interleave \+ Encoding:\s*([\d.]+) s", comp_result.stdout)
            die_time = float(match_die.group(1)) if match_die else None
            match_lossless = re.search(r"Lossless:\s*([\d.]+) s", comp_result.stdout)
            lossless_time = float(match_lossless.group(1)) if match_lossless else None
            match_serial = re.search(r"Serialization:\s*([\d.]+) s", comp_result.stdout)
            serialization_time = float(match_serial.group(1)) if match_serial else None

            match_idie = re.search(r"Decoding \+ Reposition \+ Recompose:\s*([\d.]+) s", decmp_result.stdout)
            idie_time = float(match_idie.group(1)) if match_idie else None
            match_ilossless = re.search(r"Lossless:\s*([\d.]+) s", decmp_result.stdout)
            ilossless_time = float(match_ilossless.group(1)) if match_ilossless else None
            match_iserial = re.search(r"Deserialization:\s*([\d.]+) s", decmp_result.stdout)
            iserialization_time = float(match_iserial.group(1)) if match_iserial else None
            comp_th =  total_bytes / (serialization_time + die_time + lossless_time) / 1024 / 1024 / 1024
            dec_th = total_bytes / (iserialization_time + idie_time + ilossless_time) / 1024 / 1024 / 1024
            # print(total_bytes / float(additional_bytes[i]),  comp_th, dec_th)
            if i != 0:
                additional_bytes[i] = float(additional_bytes[i - 1]) + float(additional_bytes[i])
            out_row = [compressor, sys.argv[1], e, comp_th, dec_th, 
                       total_bytes / float(additional_bytes[i]), psnr_list[i]]
            append_row(out_row)
        return 

    accumulate_residual(accumulate_file, decompressed_file, ddtype)
    acc_value = numbers
    psnr = compute_psnr(input_file, accumulate_file, ddtype, shape)
    out_row = [compressor, sys.argv[1], errors[0], acc_value[0], acc_value[1], acc_value[2], psnr[0]]
    append_row(out_row)

    compute_residual(input_file, accumulate_file, residual_file, ddtype)

    for e in errors[1:]:
        if compressor == "cuSZ":
            decompressed_file, numbers = run_cusz(shape, data_type, residual_file,  e * value_range, 'abs')
        elif compressor == "cuSZp":
            decompressed_file, numbers = run_cuszp(shape, data_type, residual_file,  e * value_range, 'abs')
        
        accumulate_residual(accumulate_file, decompressed_file, ddtype)
        psnr = compute_psnr(input_file, accumulate_file, ddtype, shape)
        acc_value = 1 / (1 / acc_value + 1 / numbers)
        out_row = [compressor, sys.argv[1], e, acc_value[0], acc_value[1], acc_value[2], psnr[0]]
        append_row(out_row)

        compute_residual(input_file, accumulate_file, residual_file, ddtype)

def test_progressive_residual(shape, data_type, input_file):
    run_progressive_residual(shape, data_type, input_file, 'PRISM')
    run_progressive_residual(shape, data_type, input_file, 'cuSZ')
    run_progressive_residual(shape, data_type, input_file, 'cuSZp')
    run_progressive_residual(shape, data_type, input_file, 'HP-MDR')
    run_progressive_residual(shape, data_type, input_file, 'cuZFP')

if __name__ == "__main__":
    input_file = sys.argv[2]
    data_type = "float"
    dim = 3
    shape = [384, 384, 256]
    Isporg = False
    if sys.argv[1] == "pressure" or sys.argv[1] == "density" or sys.argv[1] == "diffusivity":
        data_type = "double"
        shape = [384, 384, 256]
        if sys.argv[1] == "pressure":
            bit_rate = [1,3,5,5,5,5,6,4,6]
        elif sys.argv[1] == "density":
            bit_rate = [1,3,4,5,6,5,5,5,5]
        elif sys.argv[1] == "diffusivity":
            bit_rate = [2,3,5,5,6,4,5,6,4]
    elif sys.argv[1] == "CH4":
        data_type = "double"
        shape = [500, 500, 500]
        bit_rate = [1,1,1,4,3,6,4,5,5]
    elif sys.argv[1] == "QGRAUP":
        data_type = "float"
        shape = [500, 500, 100]
        bit_rate = [3,3,3,3,4,3,4,3,4]
    elif sys.argv[1] == "QC" or sys.argv[1] == "QG":
        data_type = "float"
        shape = [1200, 1200, 98]
        if sys.argv[1] == "QC":
            bit_rate = [1,2,3,3,4,3,5,2,5]
        elif sys.argv[1] == "QG":
            bit_rate = [1,2,4,2,4,3,4,3,4]
    elif sys.argv[1] == "temperature":
        data_type = "float"
        shape = [512, 512, 512]
        bit_rate = [4,3,4,3,5,2,4,3,4]

    test_progressive_residual(shape, data_type, input_file)
    