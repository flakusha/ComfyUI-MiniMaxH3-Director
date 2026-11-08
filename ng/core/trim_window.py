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


def linked_float(value):
    """A linked trim as seconds: None (unconnected upstream) means unset, not 0.

    A workflow saved before Trim start/end existed restores positionally, so the Context ID can sit
    where seconds belong: name it instead of raising the bare float() error."""
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        raise ValueError("A Trim value must be seconds, got %r; a workflow saved before Trim start/end "
                         "existed restores the Context ID here - reload it with the current version." % (value,)) from None


def trim_is_set(value):
    """A linked trim drives the card only when it is a positive number.

    A workflow saved before Trim start/end existed restores the Context ID (a string) where seconds
    belong: the string is truthy, so a truthiness guard would compare `str > int` and raise TypeError
    before linked_float is reached. Non-numbers fall back to the stored options range."""
    return isinstance(value, (int, float)) and value > 0


def resolve_window(trim_start, trim_end, options):
    """(start_s, end_s) the card really uses: a linked trim value is authoritative when it is set,
    else the card's stored options range (mapping or ContextOptions). The one owner of the link-wins
    rule - fingerprint, validate and execute all call it, so the three cannot drift."""
    start_s = linked_float(trim_start) if trim_is_set(trim_start) else options["start_s"]
    end_s = linked_float(trim_end) if trim_is_set(trim_end) else options["end_s"]
    return start_s, end_s


def fingerprint(file_stamp_text, start_s, end_s):
    """Cache-key text: the file plus both trim floats, so a linked trim change re-runs the card."""
    return "%s:%s:%s" % (file_stamp_text, float(start_s), float(end_s))
