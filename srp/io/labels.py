"""labels.py — data/labels 分層路徑(依 類別/數量/場景)。

場景名 `<cat><num>_scene####`(如 stack3_scene0001)→ 實際目錄
`data/labels/<cat>/<num>/<scene>/`(如 data/labels/stack/3/stack3_scene0001/)。

- `LABELS` 是自動分層 wrapper:既有的 `LABELS / scene / ...` 拼接與 `LABELS.glob(...)`
  **免改**,會自動導向分層路徑(讀寫皆是);漏改的檔在搬移後直接 FileNotFoundError,易抓。
- `label_dir(scene)` 供新程式直接取「某場景目錄」(分層後)。
"""
import re
from pathlib import Path

_BASE = Path(__file__).resolve().parents[2] / "data" / "labels"
_PAT = re.compile(r"^([a-z]+)(\d+)_scene")


def _split(seg):
    """seg 可能是 'stack3_scene0001' 或 'stack3_scene*/actual/x.json';回傳分層後 Path。
    只在「場景名那一段」之前插入 <cat>/<num>,場景目錄內的子路徑原樣接上。"""
    s = str(seg)
    head = s.split("/", 1)[0]
    m = _PAT.match(head)
    return (_BASE / m.group(1) / m.group(2) / s) if m else (_BASE / s)


def label_dir(scene):
    """回傳某場景的 labels 目錄(分層後 Path)。"""
    return _split(scene)


class _LabelRoot:
    """讓既有 `LABELS / scene / ...` 與 `LABELS.glob(...)` 免改就自動分層。"""

    def __truediv__(self, seg):
        return _split(seg)

    def glob(self, pattern):
        head = str(pattern).split("/", 1)[0]
        m = _PAT.match(head)
        base = (_BASE / m.group(1) / m.group(2)) if m else _BASE
        return base.glob(str(pattern))

    def __fspath__(self):
        return str(_BASE)

    def __str__(self):
        return str(_BASE)

    def __repr__(self):
        return f"LabelRoot({_BASE})"

    @property
    def base(self):
        return _BASE


LABELS = _LabelRoot()
