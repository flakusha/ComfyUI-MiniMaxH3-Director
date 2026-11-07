"""A connected AUDIO payload read without any file: the batch dim is ignored, nothing is decoded."""


def read_waveform(audio_in):
    """(channels, frames) float samples and the rate of a connected AUDIO payload."""
    try:
        rate = float(audio_in["sample_rate"])
        wave = audio_in["waveform"]
        while getattr(wave, "dim", lambda: 0)() > 2:
            wave = wave[0]
        frames = wave.shape[-1]
        channels = wave.shape[-2]
    except (KeyError, TypeError, AttributeError, IndexError, ZeroDivisionError):
        raise ValueError("The connected audio has no readable sound - reconnect its source, or pick a file on the card.") from None
    if rate <= 0 or frames <= 0 or channels <= 0:
        raise ValueError("The connected audio has no readable sound - reconnect its source, or pick a file on the card.")
    return wave, rate
