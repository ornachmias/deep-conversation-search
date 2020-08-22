import json
import os
import time

import praw
from tqdm import tqdm


# Based on https://github.com/google-research-datasets/coarse-discourse/blob/master/join_forum_data/join_forum_data.py
class ProcessReddit:
    def __init__(self, data_root, reddit_client_id, reddit_client_secret, user_agent):
        self._collected_data_dir = os.path.join(data_root, 'reddit')
        self._annotation_path = os.path.join(self._collected_data_dir, 'coarse_discourse_dataset.json')
        self._full_data_path = os.path.join(self._collected_data_dir, 'coarse_discourse_dump_reddit.json')
        self._reddit_client_id = reddit_client_id
        self._reddit_client_secret = reddit_client_secret
        self._user_agent = user_agent

    def get_full_data(self, n_topics=100):
        reddit = praw.Reddit(client_id=self._reddit_client_id,
                             client_secret=self._reddit_client_secret,
                             user_agent=self._user_agent)

        with open(self._annotation_path) as jsonfile:
            lines = jsonfile.readlines()
            dump_with_reddit = open(self._full_data_path, 'w')

            for line in tqdm(lines[:n_topics]):
                reader = json.loads(line)
                submission = reddit.submission(url=reader['url'])

                # Annotators only annotated the 40 "best" comments determined by Reddit
                submission.comment_sort = 'best'
                submission.comment_limit = 40

                post_id_dict = {}

                for post in reader['posts']:
                    post_id_dict[post['id']] = post

                try:
                    full_submission_id = 't3_' + submission.id
                    if full_submission_id in post_id_dict:
                        post_id_dict[full_submission_id]['body'] = submission.selftext

                        # For a self-post, this URL will be the same URL as the thread.
                        # For a link-post, this URL will be the link that the link-post is linking to.
                        post_id_dict[full_submission_id]['url'] = submission.url
                        if submission.author:
                            post_id_dict[full_submission_id]['author'] = submission.author.name

                    submission.comments.replace_more(limit=0)
                    for comment in submission.comments.list():
                        full_comment_id = 't1_' + comment.id
                        if full_comment_id in post_id_dict:
                            post_id_dict[full_comment_id]['body'] = comment.body
                            if comment.author:
                                post_id_dict[full_comment_id]['author'] = comment.author.name

                except Exception as e:
                    print('Error %s' % (e))

                found_count = 0
                for post in reader['posts']:
                    if not 'body' in post.keys():
                        print("Can't find %s in URL: %s" % (post['id'], reader['url']))
                    else:
                        found_count += 1

                dump_with_reddit.write(json.dumps(reader) + '\n')

                # To keep within Reddit API limits
                time.sleep(2)
