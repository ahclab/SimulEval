from argparse import ArgumentParser
from typing import Tuple

from simuleval.segmenters import GenericSegmenter
from simuleval.utils import segmenter_entrypoint


@segmenter_entrypoint
class FixedChunksSegmenter(GenericSegmenter):
    def __init__(self, args=None):
        super().__init__(args)

        self.fixed_chunk_num = args.fixed_chunk_num
        self.num_chunks = 0

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument(
            "--fixed-chunk-num",
            type=int,
            default=3,
            help="Number of chunks per segment",
        )

    def segment(self):
        is_start, start_offset, is_end, end_offset = (
            False,
            0,
            False,
            len(self.instance.current_samples),
        )

        self.num_chunks += 1
        if self.num_chunks == 1:
            is_start = True
            is_end = False
        elif self.num_chunks == self.fixed_chunk_num:
            is_start = False
            is_end = True
            self.num_chunks = 0
        else:
            is_start = False
            is_end = False
        # if len(self.instance.current_samples) == 0:
        if self.instance.is_finish_source:
            is_end = True

        # print(
        #    f"num_chunks: {self.num_chunks}, "
        #    f"is_start: {is_start}, start_offset: {start_offset}, "
        #    f"is_end: {is_end}, end_offset: {end_offset}"
        # )
        return {
            "is_start": is_start,
            "start_offset": start_offset,
            "is_end": is_end,
            "end_offset": end_offset,
        }

    def check_start(self, current_samples) -> Tuple[bool, int]:
        """
        Check if the segmenter should start a new segment.
        """
        pass

    def check_end(self, current_samples) -> Tuple[bool, int]:
        """
        Check if the segmenter should end the current segment.
        """
        pass
