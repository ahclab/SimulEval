import itertools
import os
from argparse import ArgumentParser
from typing import List

import yaml
from simuleval.evaluator.instance import Instance
from simuleval.segmenters import GenericSegmenter
from simuleval.segmenters.utils import detect_start_end, find_segments
from simuleval.utils import segmenter_entrypoint


@segmenter_entrypoint
class OracleSegmenter(GenericSegmenter):
    def __init__(self, args=None):
        super().__init__(args)

        with open(args.seg_yaml_file) as f:
            self.seg_yaml = yaml.load(f, Loader=yaml.BaseLoader)

        self.seg_groups = {}
        for wav_filename, seg_group in itertools.groupby(
            self.seg_yaml, key=lambda x: x["wav"]
        ):
            self.seg_groups[wav_filename] = list(seg_group)

        self.instance_total_samples = 0  # total samples of instance
        self.instance_total_time = 0  # total time of instance (in ms)
        self.fixed_chunk_num = 3
        self.num_chunks = 0

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument(
            "--seg-yaml-file",
            type=str,
            required=True,
            help="Oracle yaml file for segmentation.",
        )

    def generate_oracle_segmentation(self):
        # (1) generate a list of "0"s with length of total samples of 0 to last segment's end time
        end_time_samples = int(
            (
                float(self.seg_groups[self.wav_filename][-1]["offset"])
                + float(self.seg_groups[self.wav_filename][-1]["duration"])
            )
            * self.instance.sample_rate
        )
        self.oracle_segment_samples = [0] * int(end_time_samples)

        # (2) for each oracle segment, set the corresponding samples to 1
        for seg in self.seg_groups[self.wav_filename]:
            start_time_samples = int(float(seg["offset"]) * self.instance.sample_rate)
            end_time_samples = int(
                (float(seg["offset"]) + float(seg["duration"]))
                * self.instance.sample_rate
            )
            self.oracle_segment_samples[start_time_samples:end_time_samples] = [1] * (
                end_time_samples - start_time_samples
            )

    def reset(self):
        self.instance_total_samples = 0
        self.instance_total_time = 0
        self.generate_oracle_segmentation()

    def set_instance(self, instance: Instance):
        self.instance = instance
        self.wav_filename = os.path.basename(instance.source_wav_path)
        assert self.wav_filename in self.seg_groups
        self.reset()

    def segment(self):
        # perform segmentation
        oracle_chunk = self.oracle_segment_samples[
            self.instance_total_samples : self.instance_total_samples
            + len(self.instance.current_samples)
        ]
        segments_in_chunk = find_segments(oracle_chunk)

        # detect start and end of segments
        segment_state = detect_start_end(
            segments_in_chunk,
            len(oracle_chunk),
            self.instance.within_sent_segment,
        )

        self.instance_total_samples += len(self.instance.current_samples)
        self.instance_total_time = self.instance.len_sample_to_ms(
            self.instance_total_samples
        )

        return segment_state
