"""The shots document's manual render cuts ("render_cuts"): where the user wants a render-part boundary, and where
one is actually possible.

A cut keeps the pin and the join crossfade, like a derived Keyframe join. It snaps once to H3's grid here and is never
auto-moved afterwards: a later shot edit that leaves it in a bad spot only warns, it is the user's edit."""
from . import frames, limits
from .chunks import ChunkOpening
from .issues import Issue
from .shot_document import read_render_cuts


def _shortest(head):
    """The shortest part that can follow an edge, with this pin."""
    low, high = frames.delivered_bounds(head)
    return min(frames.delivered_candidates(low, head, low, high))


def _edge(previous, previous_head, wanted):
    """The grid frame nearest to `wanted` at which a part that started at `previous` (with that pin) can end."""
    residue, step = frames.delivered_residue(previous_head), limits.GRID_STEP
    low = previous + residue + (wanted - previous - residue) // step * step
    return min((low, low + step), key=lambda f: (abs(f - wanted), f))


def manual_joins(shots_text, total, head, cuts, joins):
    """The document's render_cuts as forced pinned joins: (frames, issues).

    `cuts`/`joins` are the derived edges already planned (project._part_edges); a manual cut snaps between them and
    keeps out of their reserved spans."""
    issues, manual = [], []
    starts = dict(cuts or {})
    for frame in (joins or ()):
        starts.setdefault(frame, ChunkOpening(head))
    for seconds in read_render_cuts(shots_text):
        wanted = frames.seconds_to_frames(seconds)
        if wanted >= total:
            issues.append(Issue("cut_beyond_end", "A manual render cut at %.2fs lies past the end of the project; "
                                "it is dropped." % seconds))
            continue
        previous = max((frame for frame in starts if frame < wanted), default=0)
        opening = starts.get(previous, ChunkOpening())
        previous_head = opening.head_frames
        external = bool(opening.context_id)
        join = _edge(previous, previous_head, wanted)
        if join in starts:
            continue          # the cut lands on an existing boundary; nothing to add
        following = min((frame for frame in starts if frame > wanted), default=total)
        fits_before = join - previous >= (max(opening.head_frames, 1) if external else _shortest(previous_head))
        fits_after = following - join >= _shortest(head)
        if not fits_before or not fits_after:
            if external and not fits_before:
                issues.append(Issue("cut_too_close", "A manual render cut at %.2fs falls inside the opening source's "
                                    "reserved prefix; it is dropped." % seconds))
            else:
                issues.append(Issue("cut_dropped", "A manual render cut at %.2fs is too close to the neighbouring "
                                    "render parts for a part of its own; it is dropped." % seconds))
            continue
        manual.append(join)
        starts[join] = ChunkOpening(head)
        if join != wanted:
            issues.append(Issue("cut_moved", "A manual render cut sits at %.2fs instead of %.2fs: render parts end "
                                "on H3's grid." % (frames.frames_to_seconds(join), frames.frames_to_seconds(wanted))))
    return tuple(manual), issues
