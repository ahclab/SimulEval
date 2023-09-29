from argparse import Namespace, ArgumentParser
from typing import Optional, Tuple
from simuleval.data.segments import SpeechSegment, TextSegment
from simuleval.utils import segmenter_entrypoint
from simuleval.evaluator.instance import Instance


class GenericSegmenter:
    """
    Generic Segmenter class.
    """

    def __init__(self, args: Optional[Namespace] = None) -> None:
        self.stream_history = []
        self.translations = []

        self.in_segment = False

    @staticmethod
    def add_args(parser: ArgumentParser):
        """
        Add arguments to parser.
        """
        pass

    def set_instance(self, instance: Instance):
        self.instance = instance

    @classmethod
    def from_args(cls, args):
        return cls(args)

    def update(self):
        pass

    def segment(self):
        if self.instance.within_sent_segment:
            # check if the segmenter should end the current segment
            is_end, end_offset = self.check_end(self.instance.current_samples)
            # check the case where start comes after end within a chunk
            # (e.g. [...e|...|s...])
            if is_end:
                is_start, start_offset = self.check_start(self.instance.current_samples)
        else:
            # check if the segmenter should start a new segment
            is_start, start_offset = self.check_start(self.instance.current_samples)
            # check the case where end comes after start within a chunk
            # (e.g. [...|s...e|...])
            if is_start:
                is_end, end_offset = self.check_end(self.instance.current_samples)

        return is_start, start_offset, is_end, end_offset

    def check_start(self, current_samples) -> Tuple[bool, int]:
        """
        Check if the segmenter should start a new segment.
        """
        assert NotImplementedError

    def check_end(self, current_samples) -> Tuple[bool, int]:
        """
        Check if the segmenter should end the current segment.
        """
        assert NotImplementedError