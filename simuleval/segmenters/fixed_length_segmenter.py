from argparse import ArgumentParser

from simuleval.evaluator.instance import Instance
from simuleval.segmenters import GenericSegmenter
from simuleval.utils import segmenter_entrypoint


@segmenter_entrypoint
class FixedLengthSegmenter(GenericSegmenter):
    def __init__(self, args=None):
        super().__init__(args)

        self.segment_total_samples = 0  # total samples of segment
        self.segment_total_time = 0  # total time of segment (in ms)

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument(
            "--fixed-length",
            type=int,
            required=True,
            help="Fixed length for segmentation in milliseconds.",
        )

    def reset(self):
        self.segment_total_samples = 0
        self.segment_total_time = 0
        # self.generate_fixed_segmentation()

    def set_instance(self, instance: Instance):
        self.instance = instance
        self.reset()

    def segment(self):
        is_start, start_offset, is_end, end_offset = (
            False,
            0,
            False,
            len(self.instance.current_samples),
        )

        if self.segment_total_time == 0:
            is_start = True
            start_offset = 0

        # perform fixed-length segmentation
        self.segment_total_samples += len(self.instance.current_samples)
        self.segment_total_time = self.instance.len_sample_to_ms(
            self.segment_total_samples
        )

        if self.segment_total_time >= self.args.fixed_length:
            is_end = True
            end_offset = len(self.instance.current_samples) - (
                self.segment_total_samples
                - self.instance.len_ms_to_samples(self.args.fixed_length)
            )

            is_start = True
            start_offset = end_offset + 1
            if start_offset < len(self.instance.current_samples):
                self.segment_total_samples = (
                    len(self.instance.current_samples) - start_offset
                )
                self.segment_total_time = self.instance.len_sample_to_ms(
                    self.segment_total_samples
                )
            else:
                is_start = False
                start_offset = 0
                self.segment_total_samples = 0
                self.segment_total_time = 0

        return {
            "is_start": is_start,
            "start_offset": start_offset,
            "is_end": is_end,
            "end_offset": end_offset,
        }
