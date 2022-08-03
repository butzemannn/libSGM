#!/bin/env python3

import os
import numpy as np
import argparse
import xml.etree.ElementTree as ET


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A tool to generate calib files for libSGM from KITTI calibration file format.")
    parser.add_argument("source_folder", type=str, help="The folder containing the KITTI calibration files to be converted.")
    parser.add_argument("result_folder", type=str, help="The folder for the resulting calibration files.")
    return parser


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def load_calib_file(file_location: str) -> np.ndarray:
    with open(file_location, 'r') as f:
        lines = f.readlines()
        for line in lines:
            lc = line.split(' ')
            if not lc[0] == "P2:":
                # only camera 2 data is needed
                continue
            matrix = np.array([
                [lc[1], lc[2],  lc[3],  lc[4]],
                [lc[5], lc[6],  lc[7],  lc[8]],
                [lc[9], lc[10], lc[11], lc[12][:-2]]
                ])

            return matrix


def save_calib_file(
        file_location: str, 
        fx: np.float32, 
        fy: np.float32, 
        cx: np.float32, 
        cy: np.float32, 
        p0t: np.float32, 
        p1t: np.float32
        ) -> None:
    root = ET.Element("opencv_storage")
    flx = ET.SubElement(root, "FocalLengthX")
    fly = ET.SubElement(root, "FocalLengthY")
    ctx = ET.SubElement(root, "CenterX")
    cty = ET.SubElement(root, "CenterY")
    p0 = ET.SubElement(root, "P0")
    p1 = ET.SubElement(root, "P1")
    bl = ET.SubElement(root, "BaseLine")
    h = ET. SubElement(root, "Height")
    t = ET.SubElement(root, "Tilt")
    
    flx.text = fx
    fly.text = fy
    ctx.text = cx
    cty.text = cy
    p0.text = p0t
    p1.text = p1t
    bl.text = "0.54"
    h.text = "1.65"
    t.text = "0."

    tree = ET.ElementTree(root)
    tree.write(file_location, encoding="utf-8", xml_declaration=True)

            

def generate_calib_files(source_folder: str, result_folder: str) -> None:
    if not os.path.isdir(source_folder):
        raise TypeError("The source folder must be a folder")

    if not os.path.isdir(result_folder):
        raise TypeError("The result folder must be a folder")

    calib_files = os.listdir(source_folder)
    printProgressBar(0, len(calib_files), prefix='Progress:', suffix='Complete', length=50)
    for i, calib_file in enumerate(calib_files):
        matrix = load_calib_file(f"{source_folder}/{calib_file}")
        save_calib_file(f"{result_folder}/{calib_file[:-4]}.xml", matrix[0,0], matrix[1,1], matrix[0,2], matrix[1,2], matrix[0,3], matrix[1,3])
        printProgressBar(i+1, len(calib_files), prefix='Progress:', suffix='Complete', length=50)


def main():
    parser = setup_parser()
    args = parser.parse_args()
    generate_calib_files(args.source_folder, args.result_folder)


if __name__ == "__main__":
    main()
