"""Label texts: the format of every registered number is "name: description", the name alone when there is no
description. Mode "noex" removes quoted example phrasings from the description with the data notebook's
strip_examples (the block between the markers below is a verbatim copy and must stay byte-identical)."""
import re

MODES = ("v2", "v3s", "noex", "name")


# ---- verbatim copy (the data notebook, sections 2 and 9) ----
def tokens(s):
    return re.findall(r"[a-z0-9]+", str(s).lower())


def norm_text(t):
    return " ".join(tokens(t))


QUOTE_RE = re.compile(r'["“]([^"“”]{3,})["”]')
LEAD = (r"(?:for example|for instance|e\.g\.|i\.e\.|such as|like|including|"
        r"(?:example )?(?:phrasings?|examples?|phrases?)(?: include| are| like| could be| might be)?|"
        r"(?:users? )?(?:might|may|could|would) (?:say|ask|write|type)|as in)")

def strip_examples(d, label=""):
    # Remove quoted example phrasings (not the quoted label name) and the glue words around them
    lab = norm_text(label)
    def sub(m):
        return m.group(0) if lab and norm_text(m.group(1)) == lab else " \x00 "
    s = re.sub(r'["“]([^"“”]{1,})["”]', sub, str(d))
    s = re.sub(r"\x00(?:\s*[,;.]?\s*(?:(?:or|and)\s+)?\x00)*", "\x00", s)
    s = re.sub(r"[,;:]?\s*\(?\s*\b" + LEAD + r"\s*[:,]?\s*\x00\s*\)?", "\x00", s, flags=re.I)
    s = re.sub(r"\s*\(\s*\x00\s*\)", "", s)
    s = re.sub(r"[,;:]?\s*\x00\s*[,;:]?\s*(?=[A-Z])", ". ", s)
    s = re.sub(r"[,;:]?\s*\x00\s*", " ", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"([,;:])\s*([.!?])", r"\2", s)
    s = re.sub(r"([.!?])\s*\.", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip(" ,;:")
    if s and s[-1] not in '.!?"':
        s += "."
    return s
# ---- end of the verbatim copy ----


def label_text_of(name, desc):
    # Label text format of step 0 and the step-1 data: "name: description", name only without a description
    return f"{name}: {desc}" if desc else name


def parse_labels(labels):
    """labels: a list of strings (names) or of dicts with "name" and an optional "description" (or (name, description)
    pairs). Returns (names, descriptions) with surrounding whitespace removed."""
    if isinstance(labels, (str, bytes, dict)):
        raise TypeError("labels must be a list of names or of {'name', 'description'} dicts")
    names, descs = [], []
    for i, x in enumerate(labels):
        if isinstance(x, str):
            name, desc = x, ""
        elif isinstance(x, dict):
            if "name" not in x:
                raise ValueError(f"label {i}: a dict needs a 'name' key")
            name, desc = x["name"], x.get("description") or ""
        elif isinstance(x, (tuple, list)) and len(x) == 2:
            name, desc = x[0], x[1] or ""
        else:
            raise TypeError(f"label {i}: expected a name, a dict or a (name, description) pair, got {type(x).__name__}")
        name, desc = str(name).strip(), str(desc).strip()
        if not name:
            raise ValueError(f"label {i}: empty name")
        names.append(name)
        descs.append(desc)
    if not names:
        raise ValueError("no labels")
    return names, descs


def label_texts(names, descs, mode="v2"):
    """The texts the model sees for each label in the given mode."""
    if mode in ("v2", "v3s"):
        return [label_text_of(n, d) for n, d in zip(names, descs)]
    if mode == "noex":
        return [label_text_of(n, strip_examples(d, n) if d else "") for n, d in zip(names, descs)]
    if mode == "name":
        return list(names)
    raise ValueError(f"label_text_mode {mode!r}: expected one of {MODES}")
