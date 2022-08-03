#!/bin/env python3

import os
import numpy as np
import argparse
from numba import jit

import kitti_util


def parse_files_kitti(csv_folder: str, result_folder: str) -> None:
    """
    Parses csv files to match kitti binary files. The file name will be the same except the ending, which will be changed to .bin.
    
    :param csv_folder: The folder containing csv files to parse
    :param result_folder: Folder in which the resulting files should be stored.
    :returns: None
    """
    if not os.path.isdir(csv_folder):
        raise TypeError("The csv folder must be a folder")

    if not os.path.isdir(result_folder):
        raise TypeError("The result folder must be a folder")

    csv_files = os.listdir(csv_folder)
    l = len(csv_files)
    printProgressBar(0, l, prefix="Progress:", suffix="Complete", length=50)
    for i, csv_file in enumerate(sorted(csv_files)):
        data = np.genfromtxt(f"{csv_folder}/{csv_file}", delimiter=';', dtype="float32")
        # set reflectance as 1 by default
        # data = convert_to_velodyne_coordinates(data)
        data = convert_to_velodyne_coordinates(data)
        reflectance = np.ones((data.shape[0], 1))
        data = np.ascontiguousarray(np.concatenate((data, reflectance), axis=1))
        data = data.reshape([data.shape[0] * data.shape[1]])
        data = data.astype("float32")
        data.tofile(f"{result_folder}/{csv_file[:-4]}.bin")
        printProgressBar(i, l, prefix="Progress:", suffix="Complete", length=50)


def convert_to_velodyne_coordinates(pointcloud: np.ndarray):
    """
    Convert Point Cloud from camera coordinates into velodyne coordinates
    """
    calib = kitti_util.Calibration("/media/data/datasets/kitti/training/calib/000001.txt")
    pointcloud = calib.project_rect_to_velo(pointcloud)

    return pointcloud


def setup_parser() -> argparse.ArgumentParser:
    """
    Setup the argument parser for command line arguments
    """
    parser = argparse.ArgumentParser(description="A tool to convert csv files into numpy binary files which can be read by deep learning libraries.")
    parser.add_argument("csv_folder", type=str, help="The folder containing the csv files to parse.")
    parser.add_argument("result_folder", type=str, help="The folder for the resulting binary files.")
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


def main():
    parser = setup_parser()
    args = parser.parse_args()
    parse_files_kitti(args.csv_folder, args.result_folder)
    


if __name__ == "__main__":
    main()
