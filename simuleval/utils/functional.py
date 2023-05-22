# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import closing
import socket
from typing import List
import soundfile as sf
import numpy as np

def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def concat_itts_audios(
    audio_path_list:List[str],
    offset_list:List[float],
    delay:float,
    mergin:float=0.1,
) -> np.ndarray:
    
    assert len(audio_path_list) == len(offset_list)
    
    audio_list = []
    sampling_rate = -1
    for audio_path in audio_path_list:
        audio, sampling_rate = sf.read(audio_path)
        audio_list.append(audio)
    
    offset_lengths = [int(sampling_rate * (offset + delay)) for offset in offset_list]
    mergin_length = int(sampling_rate * mergin)
    
    concat_audio_list = []
    total_length = 0
    for audio, offset in zip(audio_list, offset_lengths):
        if total_length < offset:
            padding_length = total_length - offset
            zero_audio = np.zeros(padding_length)
            
            total_length += padding_length
            concat_audio_list.append(zero_audio)
        elif total_length > offset:
            zero_audio = np.zeros(mergin_length)
            
            total_length += mergin_length
            concat_audio_list.append(zero_audio)
        
        total_length += audio.shape[0]
        concat_audio_list.append(audio)
    
    concat_audio = np.concatenate(concat_audio_list)
    return concat_audio