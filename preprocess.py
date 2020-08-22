import argparse

from data_collection.process_reddit import ProcessReddit

parser = argparse.ArgumentParser(description='Deep Conversation Search Preprocess')
parser.add_argument('-d', '--data_dir', default='./data')
parser.add_argument('-i', '--reddit_client_id', required=True)
parser.add_argument('-s', '--reddit_client_secret', required=True)
parser.add_argument('-n', '--reddit_topics', type=int, required=True)
args = parser.parse_args()

reddit_collector = ProcessReddit(data_root=args.data_dir,
                                 reddit_client_id=args.reddit_client_id,
                                 reddit_client_secret=args.reddit_client_secret,
                                 user_agent='Deep Conversation Search Preprocess')

reddit_collector.get_full_data(n_topics=args.reddit_topics)

