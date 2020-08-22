import argparse

parser = argparse.ArgumentParser(description='Deep Conversation Search Analyze')
parser.add_argument('-d', '--data_dir', default='./data')
parser.add_argument('-s', '--reddit_client_secret', required=True)
parser.add_argument('-n', '--reddit_topics', type=int, required=True)
args = parser.parse_args()

