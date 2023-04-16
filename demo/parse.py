import argparse

def parse_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', nargs='?', default='localhost', type=str)
    parser.add_argument('--api_key', nargs='?', default=None, type=str) 

    args = parser.parse_args()

    return args.host, args.api_key