from typing import List, Tuple


def find_segments(lst: List):
    """Find segments of consecutive 1s in a list of 0s and 1s.

    Args:
        lst (List): A list of integers (0s and 1s).

    Returns:
        List: A list of tuples (start, end) of segments.

    Examples:
        >>> find_segments([0, 0, 0, 1, 1, 1, 0, 1, 1, 0])
        [(3, 5), (7, 8)]
        >>> find_segments([1, 1, 1])
        [(0, 2)]
        >>> find_segments([0, 0, 0])
        []
    """
    segments = []
    start = None
    for i in range(len(lst)):
        # Check for the start of a segment
        if lst[i] == 1 and start is None:
            start = i
        # Check for the end of a segment
        elif lst[i] == 0 and start is not None:
            segments.append((start, i - 1))
            start = None

    # Check if the last element is the end of a segment
    if start is not None:
        segments.append((start, len(lst) - 1))

    return segments


def detect_start_end(
    segments: List[Tuple[int, int]],
    chunk_len: int,
    within_sent_segment: bool,
    min_segment_length_outframes: int = 0,
    current_segment_length_outframes: int = 0,
) -> dict:
    """Detect start and end of segments.

    Args:
        segments (List[Tuple[int, int]]):
            List of (start, end) of segments detected by pthr_with_chunk
        chunk_len (int):
            chunk length
        within_sent_segment (bool):
            whether the current state is within a sentence segment
        min_segment_length_outframes (int, optional):
            minimum segment length in output space. Defaults to 0.

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
    {'is_start': False, 'start_offset': 0, 'is_end': False, 'end_offset': 0, 'case': 'A', 'min_detected': False}

    >>> segments = [(1, 2), (4, 6), (7, 7)]
    >>> detect_start_end(segments, chunk_len, within_sent_segment)
    {'is_start': True, 'start_offset': 1, 'is_end': True, 'end_offset': 7, 'case': "A'", 'min_detected': False}

    >>> segments = [(2, 9)]
    >>> detect_start_end(segments, chunk_len, within_sent_segment)
    {'is_start': True, 'start_offset': 2, 'is_end': False, 'end_offset': 0, 'case': 'C', 'min_detected': False}

    >>> segments = [(2, 3), (4,7), (8, 9)]
    >>> detect_start_end(segments, chunk_len, within_sent_segment)
    {'is_start': True, 'start_offset': 2, 'is_end': False, 'end_offset': 0, 'case': "C'", 'min_detected': False}

    >>> segments = [(0, 2), (4, 6), (7, 7)]
    >>> detect_start_end(segments, chunk_len, within_sent_segment)
    {'is_start': True, 'start_offset': 0, 'is_end': True, 'end_offset': 7, 'case': "A' (edge case)", 'min_detected': False}

    >>> segments = [(0, 9)]
    >>> detect_start_end(segments, chunk_len, within_sent_segment)
    {'is_start': True, 'start_offset': 0, 'is_end': False, 'end_offset': 0, 'case': 'C (edge case)', 'min_detected': False}

    >>> segments = [(0, 3), (4,7), (8, 9)]
    >>> detect_start_end(segments, chunk_len, within_sent_segment)
    {'is_start': True, 'start_offset': 0, 'is_end': False, 'end_offset': 0, 'case': "C' (edge case)", 'min_detected': False}

    >>> within_sent_segment = True
    >>> segments = [(0, 9)]
    >>> detect_start_end(segments, chunk_len, within_sent_segment)
    {'is_start': False, 'start_offset': 0, 'is_end': False, 'end_offset': 0, 'case': 'B', 'min_detected': False}

    >>> segments = [(0, 3), (4,7), (8, 9)]
    >>> detect_start_end(segments, chunk_len, within_sent_segment)
    {'is_start': True, 'start_offset': 4, 'is_end': True, 'end_offset': 3, 'case': "B'", 'min_detected': False}

    >>> segments = [(0, 7)]
    >>> detect_start_end(segments, chunk_len, within_sent_segment)
    {'is_start': False, 'start_offset': 0, 'is_end': True, 'end_offset': 7, 'case': 'D', 'min_detected': False}

    >>> segments = [(0, 2), (4, 6), (7, 7)]
    >>> detect_start_end(segments, chunk_len, within_sent_segment)
    {'is_start': False, 'start_offset': 0, 'is_end': True, 'end_offset': 7, 'case': "D'", 'min_detected': False}

    >>> segments = [(2, 3), (4,7), (8, 9)]
    >>> detect_start_end(segments, chunk_len, within_sent_segment)
    {'is_start': True, 'start_offset': 2, 'is_end': True, 'end_offset': 0, 'case': "B' (edge case)", 'min_detected': False}

    >>> segments = []
    >>> detect_start_end(segments, chunk_len, within_sent_segment)
    {'is_start': False, 'start_offset': 0, 'is_end': True, 'end_offset': 0, 'case': 'D (edge case)', 'min_detected': False}

    >>> segments = [(1, 2), (4, 6), (7, 7)]
    >>> detect_start_end(segments, chunk_len, within_sent_segment)
    {'is_start': False, 'start_offset': 0, 'is_end': True, 'end_offset': 7, 'case': "D' (edge case)", 'min_detected': False}

    """
    start_offset, end_offset = 0, 0
    case = ""  # for testing
    min_detected = False

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

    # ignore segments shorter than min_segment_length_outframes
    if case in ["B'", "D", "D'"]:
        if current_segment_length_outframes + end_offset < min_segment_length_outframes:
            # breakpoint()
            is_end = False
            end_offset = 0
            is_start = False
            start_offset = 0
            min_detected = True
    if case in ["A'"]:
        if end_offset - start_offset < min_segment_length_outframes:
            # breakpoint()
            is_end = False
            end_offset = 0
            min_detected = True

    return {
        "is_start": is_start,
        "start_offset": start_offset,
        "is_end": is_end,
        "end_offset": end_offset,
        "case": case,
        "min_detected": min_detected,
    }
