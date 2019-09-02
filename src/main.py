#!/usr/bin/env python3

import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    return parser.parse_args()

def main():
    args = parse_args()

    print('Hello World!')

if __name__ == '__main__':
    main()
