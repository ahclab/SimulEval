import math
from simuleval.segmenters import Segmenter

class FixedSegmenter(Segmenter):

    def __init__(self, args):
        super().__init__(args)

        self.segment_length = args.fixed_length
        self.speech_chunk_size = args.chunk_size

        self.n_chunks_in_segment = math.ceil(self.segment_length / self.speech_chunk_size)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--fixed-length', type=int, default=20000,
                            help='segment length (in ms)')
        return parser

    def is_segment(self, states):
        if (states.segments.source.length() + 1) % self.n_chunks_in_segment == 0:
            return False
        else:
            return True
