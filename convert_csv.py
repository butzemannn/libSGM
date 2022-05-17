#!/bin/env python3

import os
import numpy as np
import argparse
from numba import jit


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
    for csv_file in csv_files:
        data = np.genfromtxt(f"{csv_folder}/{csv_file}", delimiter=';')
        # set reflectance as 1 by default
        reflectance = np.ones((data.shape[0],1))
        data = np.concatenate((data, reflectance), axis=1)
        data = data.reshape([data.shape[0] * data.shape[1]])
        data = data.astype("float32")
        data.tofile(f"{result_folder}/{csv_file[:-4]}.bin")


def setup_parser() -> argparse.ArgumentParser:
    """
    Setup the argument parser for command line arguments
    """
    parser = argparse.ArgumentParser(description="A tool to convert csv files into numpy binary files which can be read by deep learning libraries.")
    parser.add_argument("csv_folder", type=str, help="The folder containing the csv files to parse.")
    parser.add_argument("result_folder", type=str, help="The folder for the resulting binary files.")
    return parser


def main():
    parser = setup_parser()
    args = parser.parse_args()
    parse_files_kitti(args.csv_folder, args.result_folder)
    


if __name__ == "__main__":
    main()
