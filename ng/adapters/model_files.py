"""Every model file the pack uses: H3's own (loaded by ComfyUI's loaders; listed so the Director can show what is there and
where to download it) and the extra ones our nodes load (SAM3, Qwen Image, Whisper, ...). One table, js/ng/models.json, which
the cards' Models menus read as well. A purpose takes the file the user chose in a node's Models menu when that file is there, else
the first file that fits one of its patterns (the official name first), else None: every caller has its own fallback."""
import json
import os
import re

import folder_paths

PATH = os.path.join(os.path.dirname(__file__), "..", "..", "js", "ng", "models.json")
with open(PATH, encoding="utf-8") as file:
    MODELS = json.load(file)


def names(purpose):
    """The files of the purpose's folder. When ComfyUI knows the folder - its own under models/, or one registered
    through extra_model_paths.yaml - its list is used, so files on a registered extra path show up too. Whisper's,
    MoGe's and Depth Anything 3's folders are not always registered (asking folder_paths for one that is not raised
    and the Models menu stayed empty, GUI test 2026-10-19): then the folder under models/ is read.
    A purpose with `"dirs": true` (HF model folders like Qwen3-ASR) lists directories, not files."""
    folder = MODELS[purpose]["folder"]
    want_dirs = bool(MODELS[purpose].get("dirs"))
    if folder in folder_paths.folder_names_and_paths:
        listed = folder_paths.get_filename_list(folder)
        if not want_dirs:
            return listed
        roots = folder_paths.get_folder_paths(folder)
        return [name for name in listed
                if any(os.path.isdir(os.path.join(root, name)) for root in roots)]
    root = os.path.join(folder_paths.models_dir, folder)
    if not os.path.isdir(root):
        return []
    is_wanted = os.path.isdir if want_dirs else os.path.isfile
    return sorted(name for name in os.listdir(root) if is_wanted(os.path.join(root, name)))


def find(purpose, chosen=""):
    files = names(purpose)
    if chosen and chosen in files:
        return chosen
    return next((name for pattern in MODELS[purpose]["patterns"] for name in files if re.search(pattern, name, re.I)), None)


def missing(purpose):
    """One sentence naming the file, its folder and where it comes from."""
    model = MODELS[purpose]
    where = (" (%s)" % model["url"]) if model["url"] else ""
    return "%s needs %s in models/%s%s." % (model["label"], model["file"], model["folder"], where)


def chosen(text):
    """The {purpose: file} a node's Models menu stored, from its JSON text."""
    try:
        value = json.loads(text) if text else {}
    except ValueError:
        return {}
    return {key: file for key, file in value.items() if key in MODELS and isinstance(file, str)} if isinstance(value, dict) else {}
