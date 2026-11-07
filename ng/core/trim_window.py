"""The one trim window: (Trim start, Trim end) of a source card in seconds, and the numbers derived from it.

Seconds only: the frame math stays in audio_timing (its Studio agreement is pinned byte for byte), and the LRC
shift into story time stays in lyrics (story_stamps) - windows are media-neutral, lyrics are audio-only."""


def window_seconds(start_s, end_s, length_s):
    """(start, end) the card really plays; end is None when neither the card's End nor the file names one."""
    start = max(0.0, float(start_s))
    ends = [value for value in (float(end_s), float(length_s)) if value > 0]
    return start, min(ends) if ends else None


def max_duration(start_s, end_s, length_s):
    """The playable seconds of the window; 0.0 when even the file's length is unknown."""
    start, end = window_seconds(start_s, end_s, length_s)
    return max(0.0, end - start) if end is not None else 0.0


def check_window(start_s, end_s, length_s):
    """(start, end) with the impossible windows named: inverted (the card's End before its Start) and a start at or
    past the file's end. end never lies past the file."""
    start, end = window_seconds(start_s, end_s, length_s)
    if length_s > 0 and start >= length_s:
        raise ValueError("Trim start %.2f s is at or past the file's end (%.2f s)." % (start, length_s))
    if end is not None and end <= start:
        raise ValueError("Source end must be later than source start; 0 means the file end.")
    return start, min(end, length_s) if end is not None and length_s > 0 else end


def fingerprint(file_stamp_text, start_s, end_s):
    """Cache-key text: the file plus both trim floats, so a linked trim change re-runs the card."""
    return "%s:%s:%s" % (file_stamp_text, float(start_s), float(end_s))
