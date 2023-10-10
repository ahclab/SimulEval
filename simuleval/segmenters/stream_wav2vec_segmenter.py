import logging
from argparse import ArgumentParser, Namespace
from typing import List, Tuple

import numpy as np
import torch
from simuleval.evaluator.instance import Instance
from simuleval.segmenters import GenericSegmenter
from simuleval.utils import segmenter_entrypoint
from wav2vec_segmenter.constants import INPUT_SAMPLE_RATE, TARGET_SAMPLE_RATE
from wav2vec_segmenter.utils.load_model import load_model

logger = logging.getLogger("simuleval.sentence_level_evaluator")


def pthr_with_chunk(probs: np.ndarray, threshold: float = 0.5) -> List[Tuple[int, int]]:
    """Identify segments where probabilities are above threshold.

    Args:
        probs (np.ndarray): A sequence of probabilities indicating the likelihood
            that a given frame is within a speech segment.
        threshold (float, optional): Probability threshold. Only segments with
            probabilities exceeding this value will be selected. Defaults to 0.5.

    Returns:
        List[Tuple[int, int]]: A list of tuples where each tuple represents
        the start and end position of an audio segment.

    Examples:
        # case (A): セグメント外 -> 維持
        >>> probs = np.array([0, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1])
        >>> pthr_with_chunk(probs, threshold=0.5)
        []

        # case (A'): セグメント外 -> 立ち上がり (セグメントに入る) -> ... -> 立ち下がり (セグメントから出る)
        >>> probs = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 0])
        >>> pthr_with_chunk(probs, threshold=0.5)
        [(2, 3), (5, 5), (7, 7)]

        # case (B): セグメント内 -> 維持
        >>> probs = np.array([0.8, 0.9, 0.9, 1, 1, 1])
        >>> pthr_with_chunk(probs, threshold=0.5)
        [(0, 5)]

        # case (B'): セグメント内 -> 立ち下がり (セグメントから出る) -> ... -> 立ち上がり (セグメントに入る)
        >>> probs = np.array([1, 1, 1, 0, 1, 0, 0, 0, 1, 1])
        >>> pthr_with_chunk(probs, threshold=0.5)
        [(0, 2), (4, 4), (8, 9)]

        # case (C)(C') セグメント外 -> ... -> 立ち上がり
        >>> probs = np.array([0, 0, 0, 0, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6])
        >>> pthr_with_chunk(probs, threshold=0.5)
        [(4, 9)]
        >>> probs = np.array([0, 0, 0, 0, 0.6, 0.6, 0, 0.6, 0, 0.6])
        >>> pthr_with_chunk(probs, threshold=0.5)
        [(4, 5), (7, 7), (9, 9)]

        # cased (D)(D') セグメント内 -> ... -> 立ち下がり
        >>> probs = np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0, 0, 0, 0])
        >>> pthr_with_chunk(probs, threshold=0.5)
        [(0, 5)]
        >>> probs = np.array([0.6, 0.6, 0, 0.6, 0, 0.6, 0, 0, 0, 0])
        >>> pthr_with_chunk(probs, threshold=0.5)
        [(0, 1), (3, 3), (5, 5)]

    """
    segments = []
    start_pos = None

    for i, prob in enumerate(probs):
        # Check if we are entering a segment
        if prob > threshold and start_pos is None:
            start_pos = i
        # Check if we are leaving a segment
        elif prob <= threshold and start_pos is not None:
            segments.append((start_pos, i - 1))
            start_pos = None

    # Check if the last segment goes till the end
    if start_pos is not None:
        segments.append((start_pos, len(probs) - 1))

    return segments


def detect_start_end(
    segments: List[Tuple[int, int]],
    chunk_len: int,
    within_sent_segment: bool,
) -> dict:
    """Detect start and end of segments.

    Args:
        segments (List[Tuple[int, int]]):
            List of (start, end) of segments detected by pthr_with_chunk
        chunk_len (int):
            chunk length
        within_sent_segment (bool):
            whether the current state is within a sentence segment

    Returns:
        dict: A dictionary containing the following keys:
            - is_start (bool): whether the current chunk is the start of a segment
            - start_offset (int): start offset of the segment
            - is_end (bool): whether the current chunk is the end of a segment
            - end_offset (int): end offset of the segment
            - case (str): case of the segment

    Examples:
    >>> chunk_len = 10
    >>> within_sent_segment = False
    >>> segments = []
    >>> detect_start_end(segments, chunk_len, within_sent_segment)
    {'is_start': False, 'start_offset': 0, 'is_end': False, 'end_offset': 0, 'case': 'A'}

    >>> segments = [(1, 2), (4, 6), (7, 7)]
    >>> detect_start_end(segments, chunk_len, within_sent_segment)
    {'is_start': True, 'start_offset': 1, 'is_end': True, 'end_offset': 7, 'case': "A'"}

    >>> segments = [(2, 9)]
    >>> detect_start_end(segments, chunk_len, within_sent_segment)
    {'is_start': True, 'start_offset': 2, 'is_end': False, 'end_offset': 0, 'case': 'C'}

    >>> segments = [(2, 3), (4,7), (8, 9)]
    >>> detect_start_end(segments, chunk_len, within_sent_segment)
    {'is_start': True, 'start_offset': 2, 'is_end': False, 'end_offset': 0, 'case': "C'"}

    >>> segments = [(0, 2), (4, 6), (7, 7)]
    >>> detect_start_end(segments, chunk_len, within_sent_segment)
    {'is_start': True, 'start_offset': 0, 'is_end': True, 'end_offset': 7, 'case': "A' (edge case)"}

    >>> segments = [(0, 9)]
    >>> detect_start_end(segments, chunk_len, within_sent_segment)
    {'is_start': True, 'start_offset': 0, 'is_end': False, 'end_offset': 0, 'case': 'C (edge case)'}

    >>> segments = [(0, 3), (4,7), (8, 9)]
    >>> detect_start_end(segments, chunk_len, within_sent_segment)
    {'is_start': True, 'start_offset': 0, 'is_end': False, 'end_offset': 0, 'case': "C' (edge case)"}

    >>> within_sent_segment = True
    >>> segments = [(0, 9)]
    >>> detect_start_end(segments, chunk_len, within_sent_segment)
    {'is_start': False, 'start_offset': 0, 'is_end': False, 'end_offset': 0, 'case': 'B'}

    >>> segments = [(0, 3), (4,7), (8, 9)]
    >>> detect_start_end(segments, chunk_len, within_sent_segment)
    {'is_start': True, 'start_offset': 4, 'is_end': True, 'end_offset': 3, 'case': "B'"}

    >>> segments = [(0, 7)]
    >>> detect_start_end(segments, chunk_len, within_sent_segment)
    {'is_start': False, 'start_offset': 0, 'is_end': True, 'end_offset': 7, 'case': 'D'}

    >>> segments = [(0, 2), (4, 6), (7, 7)]
    >>> detect_start_end(segments, chunk_len, within_sent_segment)
    {'is_start': False, 'start_offset': 0, 'is_end': True, 'end_offset': 7, 'case': "D'"}

    >>> segments = [(2, 3), (4,7), (8, 9)]
    >>> detect_start_end(segments, chunk_len, within_sent_segment)
    {'is_start': True, 'start_offset': 2, 'is_end': True, 'end_offset': 0, 'case': "B' (edge case)"}

    >>> segments = []
    >>> detect_start_end(segments, chunk_len, within_sent_segment)
    {'is_start': False, 'start_offset': 0, 'is_end': True, 'end_offset': 0, 'case': 'D (edge case)'}

    >>> segments = [(1, 2), (4, 6), (7, 7)]
    >>> detect_start_end(segments, chunk_len, within_sent_segment)
    {'is_start': False, 'start_offset': 0, 'is_end': True, 'end_offset': 7, 'case': "D' (edge case)"}

    """
    start_offset, end_offset = 0, 0
    case = ""  # for testing

    # case B, B', D, D'
    if within_sent_segment:
        if len(segments) > 0:
            if segments[0][0] == 0:  # almost always True
                if segments[-1][1] == chunk_len - 1:
                    # case B
                    if len(segments) == 1:
                        case = "B"
                        is_end = False
                        is_start = False
                    # case B'
                    else:
                        case = "B'"
                        is_end = True
                        is_start = True
                        end_offset = segments[0][1]
                        start_offset = segments[1][0]
                else:
                    # case D
                    if len(segments) == 1:
                        case = "D"
                        is_end = True
                        is_start = False
                        end_offset = segments[-1][1]
                    # case D' (treat in the same way as case D)
                    else:
                        case = "D'"
                        is_end = True
                        is_start = False
                        end_offset = segments[-1][1]
            else:  # edge case where eos appears at frame 0
                # case B' (edge case)
                if segments[-1][1] == chunk_len - 1:
                    case = "B' (edge case)"
                    is_end = True
                    is_start = True
                    end_offset = 0
                    start_offset = segments[0][0]
                # case D' (edge case)
                else:
                    case = "D' (edge case)"
                    is_end = True
                    is_start = False
                    end_offset = segments[-1][1]
        # case D (edge case)
        else:
            case = "D (edge case)"
            is_end = True
            is_start = False
            end_offset = 0
    # case A, A', C, C'
    else:
        # case A
        if len(segments) == 0:
            case = "A"
            is_start = False
            is_end = False
        else:
            if segments[0][0] != 0:  # almost always True
                if segments[-1][1] == chunk_len - 1:
                    # case C
                    if len(segments) == 1:
                        case = "C"
                        is_start = True
                        is_end = False
                        start_offset = segments[0][0]
                    # case C' (treat in the same way as case C)
                    else:
                        case = "C'"
                        is_start = True
                        is_end = False
                        start_offset = segments[0][0]
                else:
                    # case A'
                    case = "A'"
                    is_start = True
                    is_end = True
                    start_offset = segments[0][0]
                    end_offset = segments[-1][1]
            else:  # edge case where bos appears at frame 0
                if segments[-1][1] == chunk_len - 1:
                    if len(segments) == 1:
                        # case C (edge case)
                        case = "C (edge case)"
                        is_start = True
                        is_end = False
                        start_offset = 0
                    else:
                        # case C' (edge case)
                        case = "C' (edge case)"
                        is_start = True
                        is_end = False
                        start_offset = 0
                else:
                    # case A' (edge case)
                    case = "A' (edge case)"
                    is_start = True
                    is_end = True
                    start_offset = 0
                    end_offset = segments[-1][1]

    return {
        "is_start": is_start,
        "start_offset": start_offset,
        "is_end": is_end,
        "end_offset": end_offset,
        "case": case,
    }


@segmenter_entrypoint
class StreamWav2VecSegmenter(GenericSegmenter):
    def __init__(self, args: Namespace | None = None) -> None:
        super().__init__(args)

        self.num_chunks = 0
        self.segment_history = None  # [past samples in current segment]
        self.chunk_history = []  # [[chunk1], [chunk2], ...]

        self.device = (
            torch.device("cuda:0")
            if torch.cuda.device_count() > 0
            else torch.device("cpu")
        )

        # build segmenter model
        self.model, _, self.config = load_model(
            args.seg_model_config_path, args.seg_model_path
        )
        self.model = self.model.to(self.device)

        self.trg_in_ratio = TARGET_SAMPLE_RATE / INPUT_SAMPLE_RATE  # 0.003121875
        self.in_trg_ratio = INPUT_SAMPLE_RATE / TARGET_SAMPLE_RATE  # 320.3203203203203

    @staticmethod
    def add_args(parser: ArgumentParser):
        parser.add_argument(
            "--seg-model-path",
            type=str,
            default="",
            help="Path to the segmentation model",
        )
        parser.add_argument(
            "--seg-model-config-path",
            type=str,
            default="",
            help="Path to the segmentation model config",
        )
        parser.add_argument(
            "--normalize-audio",
            action="store_true",
            help="Normalize audio before inference",
        )
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.5,
            help="Threshold for segmentation",
        )
        #        parser.add_argument(
        #            "--without-context",
        #            action="store_true",
        #            help="Do not use context from previous chunks",
        #        )
        parser.add_argument(
            "--context-type",
            type=str,
            choices=["segment", "fixed-chunk", "None"],
            default="segment",
            help="Type of context to use",
        )
        if parser.get_default("context_type") == "fixed-chunk":
            parser.add_argument(
                "--context-chunk-num",
                type=int,
                default=3,
                help="Number of chunks to use as context",
            )

    def reset(self):
        self.num_chunks = 0
        self.segment_history = None
        self.chunk_history = []

    def set_instance(self, instance: Instance):
        self.instance = instance
        self.reset()

    def convert_samples_list_to_tensor(self, samples: list[float]) -> torch.Tensor:
        """Convert samples list to tensor.

        Args:
            samples (list[float]): list of samples

        Returns:
            torch.Tensor: tensor of samples with shape (1, -1)

        Examples:
            >>> args = Namespace( \
                seg_model_path=None, seg_model_config_path=None) # doctest: +SKIP
            >>> c = StreamWav2VecSegmenter(args)  # doctest: +SKIP
            >>> samples = [0.1, 0.2, 0.3]  # doctest: +SKIP
            >>> c.convert_samples_list_to_tensor(samples)  # doctest: +SKIP
            tensor([[0.1000, 0.2000, 0.3000]])
            >>> c.convert_samples_list_to_tensor(  \
                [0.039306640625, 0.01202392578125]) # doctest: +SKIP
            tensor([[0.0393, 0.0120]])
        """
        return torch.tensor(samples).unsqueeze(0)

    def _inframes_to_outframes(self, x):
        # from input space to output space
        return np.round(x * self.trg_in_ratio).astype(int)

    def _outframes_to_inframes(self, x):
        # from output space to input space
        return np.round(x * self.in_trg_ratio).astype(int)

    def inference(self, audio: torch.Tensor, normalize_audio: bool) -> torch.Tensor:
        """Inference segmentation model.

        Args:
            audio (torch.Tensor): tensor of audio with shape (1, -1)
            normalize_audio (bool): whether to normalize audio with Z-score

        Returns:
            np.ndarray: segmentation probabilities with shape (1, -1)

        Example:
            >>> c = StreamWav2VecSegmenter(args)  # doctest: +SKIP
            >>> audio = torch.randn((1, 15200))  # doctest: +SKIP
            >>> c.inference(audio, normalize_audio=True)[:, :4]   # doctest: +SKIP
            array([[0.00064405, 0.00063207, 0.00063119, 0.00061347]], dtype=float32)
            >>> c.inference(audio, normalize_audio=False)[:, :4]   # doctest: +SKIP
            array([[0.00064345, 0.00062967, 0.00063069, 0.00061253]], dtype=float32)


        Notes:
            If `normalize_audio` is set to True, the audio tensor is normalized
                by subtracting the mean and dividing by the standard deviation.
            The function prepares input masks for both wav2vec and segmentation
                models. These masks are used to handle padding in the wav2vec 2.0 model.
            Sometimes, the output from the wav2vec model may have a mismatch in
                frame size. The function corrects for these cases before passing the
                hidden states to the segmentation model.
        """

        # normalize audio. See wav2vec_segmenter/datautils.py for more details.
        if normalize_audio:
            audio = (audio - torch.mean(audio)) / torch.std(audio)

        audio = audio.to(self.device)

        # input mask for wav2vec_model with all ones
        # (in wav2vec 2.0, this is used to mask padding)
        in_mask = torch.ones(audio.shape, dtype=torch.long).to(self.device)

        # input mask for seg_model with all ones
        out_mask = torch.ones(
            [1, self._inframes_to_outframes(audio.shape[1])], dtype=torch.long
        ).to(self.device)

        # inference
        with torch.no_grad():
            _, wav2vec_hidden = self.model.wav2vec_model(audio, in_mask)

            # some times the output of wav2vec is 1 frame larger/smaller
            # correct for these cases
            size1 = wav2vec_hidden.shape[1]
            size2 = out_mask.shape[1]
            if size1 != size2:
                if size1 < size2:
                    out_mask = out_mask[:, :-1]
                else:
                    wav2vec_hidden = wav2vec_hidden[:, :-1, :]

            logits = self.model.seg_model(wav2vec_hidden, out_mask)
            probs = torch.sigmoid(logits)
            probs = probs.detach().cpu().numpy()

        return probs

    def segment(self):
        audio = self.convert_samples_list_to_tensor(self.instance.current_samples)

        # add context to audio
        if self.args.context_type == "segment":
            if self.segment_history is not None:
                audio_with_context = torch.cat([self.segment_history, audio], dim=1)
            else:
                audio_with_context = audio
        elif self.args.context_type == "fixed-chunk":
            raise NotImplementedError
        else:
            audio_with_context = audio

        probs = self.inference(audio_with_context, self.args.normalize_audio)

        # remove context from probs
        if self.args.context_type == "segment":
            if self.segment_history is not None:
                probs = probs[
                    :, self._inframes_to_outframes(self.segment_history.shape[1]) :
                ]

        # perform segmentation with pthr
        segments_in_chunk = pthr_with_chunk(probs[0], threshold=self.args.threshold)

        # detect start and end of segments
        segment_state = detect_start_end(
            segments_in_chunk, len(probs[0]), self.instance.within_sent_segment
        )

        # convert start and end offsets from output space to input space
        segment_state["start_offset"] = self._outframes_to_inframes(
            segment_state["start_offset"]
        )
        segment_state["end_offset"] = self._outframes_to_inframes(
            segment_state["end_offset"]
        )

        # update input history
        if self.args.context_type == "segment":
            # case B', C, C'
            if (
                segment_state["is_start"]
                and not segment_state["end_offset"] > segment_state["start_offset"]
            ):
                self.segment_history = audio[:, segment_state["start_offset"] :]
            # case B
            elif self.instance.within_sent_segment and not segment_state["is_end"]:
                self.segment_history = torch.cat([self.segment_history, audio], dim=1)
            # case A, A', D, D'
            else:
                self.segment_history = None
        elif self.args.context_type == "fixed-chunk":
            raise NotImplementedError

        # print(segment_state)
        # if segment_state["is_start"]:
        # breakpoint()

        return segment_state
