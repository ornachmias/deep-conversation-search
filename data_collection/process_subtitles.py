import os
import re
from datetime import datetime
import csv


class ProcessSubtitles:
    def __init__(self, data_root):
        self._subs_dir = os.path.join(data_root, 'subtitles')
        self._time_threshold = 5

    def process(self):
        raw_sub_files = [(os.path.join(self._subs_dir, f), f) for f
                         in os.listdir(self._subs_dir)
                         if os.path.isfile(os.path.join(self._subs_dir, f)) and f.endswith('.srt')]

        for path, name in raw_sub_files:
            with open(path) as f:
                lines = f.readlines()
                convs = self._process_file(lines)
                self._save_conversations(name, convs)

    def _save_conversations(self, file_name, convs):
        out_path = os.path.join(self._subs_dir, file_name.split('.')[0] + '.csv')
        with open(out_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['conv_id', 'line_id', 'content'])

        for conv_id in range(len(convs)):
            for line_id in range(len(convs[conv_id])):
                line = convs[conv_id][line_id].replace(",", "")
                if line[0] == '-':
                    line = line[1:]

                with open(out_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([conv_id, line_id, line])

    def _process_file(self, file_lines):
        convs = []
        current_conv = []
        last_end_time = None

        for line in file_lines:
            line = line.strip()
            if ProcessSubtitles._is_line_metadata(line) or not line:
                continue

            if ProcessSubtitles._is_time_metadata(line):
                times = line.split(" --> ")
                cur_start_time = ProcessSubtitles._parse_datetime(times[0])
                cur_end_time = ProcessSubtitles._parse_datetime(times[-1])

                if last_end_time is not None \
                        and (cur_start_time - last_end_time).total_seconds() > self._time_threshold:
                    convs.append(current_conv)
                    current_conv = []

                last_end_time = cur_end_time

                continue

            if line.strip():
                current_conv.append(line)

        convs.append(current_conv)
        return convs

    @staticmethod
    def _parse_datetime(str_time):
        return datetime.strptime(str_time, '%H:%M:%S,%f')

    @staticmethod
    def _is_time_metadata(line):
        pattern = re.compile('\\d\\d:\\d\\d:\\d\\d,\\d* --> \\d\\d:\\d\\d:\\d\\d,\\d*')
        return pattern.match(line) is not None

    @staticmethod
    def _is_line_metadata(line):
        pattern = re.compile('\\d+$')
        return pattern.match(line) is not None


process = ProcessSubtitles('./data')
process.process()
