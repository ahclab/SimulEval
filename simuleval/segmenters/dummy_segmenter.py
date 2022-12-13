from simuleval.segmenters import Segmenter

class DummySegmenter(Segmenter):

    def __init__(self, args):
        super().__init__(args)

        self.segment_boundaries_in_ms = \
          [10000,20000,30000,40000,50000,60000,70000,80000,90000,100000]
        self.end_seg_time = self.segment_boundaries_in_ms.pop(0)

    @staticmethod
    def add_args(parser):
        return parser

    def is_segment(self, states):
        if states.num_milliseconds() >= self.end_seg_time:
            if len(self.segment_boundaries_in_ms) > 0:
                self.end_seg_time = self.segment_boundaries_in_ms.pop(0)
            return False
        else:
            return True
