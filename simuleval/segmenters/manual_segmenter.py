import yaml
import itertools
from simuleval.segmenters import Segmenter

class ManualYamlSegmenter(Segmenter):

    def __init__(self, args):
        super().__init__(args)

        self.speech_chunk_size = args.chunk_size

        with open(args.path_to_yaml) as f:
            self.segments = yaml.load(f, Loader=yaml.BaseLoader)

        self.seg_groups = []
        for wav_filename, seg_group in itertools.groupby(self.segments, lambda x: x["wav"]):
            segment_list = []
            for i, segment in enumerate(seg_group):
                segment_list.append(segment)
            self.seg_groups.append(segment_list)

        self.instance_id = 0
        self.update_segmenter(self.instance_id)

    def update_segmenter(self, instance_id):
        if len(self.seg_groups[instance_id]) > 0:
            self.curr_seg = self.seg_groups[instance_id].pop(0)
            self.start_time = float(self.curr_seg["offset"]) * 1000
            self.end_time = self.start_time + float(self.curr_seg["duration"]) * 1000
        else:
            self.curr_seg = None

    @staticmethod
    def add_args(parser):
        parser.add_argument("--path-to-yaml", "-yaml", type=str, required=True,
                            help="path to segmentation file")
        return parser

    def is_segment(self, states):
        # read first segment of an instance
        if self.instance_id != states.instance_id:
            self.instance_id = states.instance_id
            self.update_segmenter(self.instance_id)

        while states.num_milliseconds() > self.end_time:
            self.update_segmenter(states.instance_id)
            if self.curr_seg == None:
                return False

        if states.num_milliseconds() >= self.start_time and \
            states.num_milliseconds() <= self.end_time:
            return True

        return False
