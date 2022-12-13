class Segmenter(object):

    def __init__(self, args):

        self.stream_history = []
        self.translations = []
        pass

    @staticmethod
    def add_args(parser):
        # Add additional command line arguments here
        pass

    def update_history(self, in_segment):
        self.stream_history.append(in_segment)

    def update_target(self, translation):
        self.translations.append(translation)

    def reset_status(self, states):
        # reset source units (encoder states)
        states.units.source.value = list()

        # reset target units (decoder states)
        states.units.target.value = list()
        states.segments.target.value = list()

        # set finish_read() = False and finish_write() = False
        states.status["read"] = True
        states.status["write"] = True

    def is_segment(self, states):
        # Make decision here
        assert NotImplementedError
