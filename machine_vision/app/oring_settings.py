"""OK / NG judgement settings of the o-ring detector: one place, shared by HomePage and SettingPage.

Stored in app/config.ini, section [oring_v2]. Every value is validated / clamped, so a hand-edited or corrupt file can
never crash the app: bad values fall back to the defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

from app import ini_store
from app.oring_v2 import Params

SECTION = "oring_v2"
ENGINES = ("v2", "legacy")
NA_SEND_CHOICES = ("NG", "NA", "OK")
SEND_PER_CHOICES = ("part", "frame")

LIMITS = {                      # (min, max)
    "dist_nominal_px": (0, 300),
    "dist_tol_px": (0, 150),
    "min_bump_px": (0.0, 50.0),
    "min_width_px": (0, 400),
}


def _num(value, default, lo, hi, cast):
    try:
        v = cast(float(str(value).strip()))
    except (TypeError, ValueError):
        return default
    return cast(min(max(v, lo), hi))


def _choice(value, choices, default):
    v = str(value).strip()
    for c in choices:
        if v.lower() == c.lower():
            return c
    return default


def _flag(value, default):
    v = str(value).strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return default


@dataclass
class OringSettings:
    engine: str = "v2"             # v2 = new detector, legacy = the old find_oring (kept as a fallback, not shown in the UI)
    dist_nominal_px: int = 55      # ring-to-tip distance the parts normally have (data of 2026-09: median 55, p10-p90 50-60)
    dist_tol_px: int = 15          # OK while nominal - tol <= distance <= nominal + tol  (40 - 70 px)
    min_bump_px: float = 4.0       # the flange must protrude at least this much (real rings 10-24 px, plain tube 0-3 px)
    min_width_px: int = 0          # minimum ring width, 0 = not checked
    na_send_as: str = "NG"         # what is sent to PLC / UDP for a part that cannot be evaluated (NA)
    send_per: str = "part"         # part = one UDP message per part, frame = one per frame (worst part)
    show_measure: bool = True      # write the measured numbers next to every ring on the result image

    # ---- derived ----
    @property
    def dist_range(self) -> tuple[int, int]:
        return max(0, self.dist_nominal_px - self.dist_tol_px), self.dist_nominal_px + self.dist_tol_px

    def clean(self) -> "OringSettings":
        """A validated copy (values clamped, unknown choices replaced by the defaults)."""
        d = OringSettings()
        return OringSettings(
            engine=_choice(self.engine, ENGINES, d.engine),
            dist_nominal_px=_num(self.dist_nominal_px, d.dist_nominal_px, *LIMITS["dist_nominal_px"], int),
            dist_tol_px=_num(self.dist_tol_px, d.dist_tol_px, *LIMITS["dist_tol_px"], int),
            min_bump_px=round(_num(self.min_bump_px, d.min_bump_px, *LIMITS["min_bump_px"], float), 2),
            min_width_px=_num(self.min_width_px, d.min_width_px, *LIMITS["min_width_px"], int),
            na_send_as=_choice(self.na_send_as, NA_SEND_CHOICES, d.na_send_as),
            send_per=_choice(self.send_per, SEND_PER_CHOICES, d.send_per),
            show_measure=_flag(self.show_measure, d.show_measure),
        )

    def to_params(self) -> Params:
        """Detector parameters: the judgement settings on top of the tested defaults of oring_v2.Params."""
        s = self.clean()
        lo, hi = s.dist_range
        p = Params()
        p.min_dist_px = float(lo)
        p.max_dist_ok_px = float(hi)
        p.max_dist_px = max(p.max_dist_px, p.max_dist_ok_px + 5.0)     # the search window must reach the OK limit
        p.min_bump_px = float(s.min_bump_px)
        p.min_width_px = float(s.min_width_px)
        p.draw_labels = bool(s.show_measure)
        return p

    def sent_value(self, classification: str) -> str:
        """What goes to PLC / UDP for a part verdict (NA is mapped as configured)."""
        c = (classification or "NA").upper()
        return self.na_send_as if c == "NA" else c

    # ---- file ----
    def to_section(self) -> dict:
        s = self.clean()
        return {f.name: ("true" if getattr(s, f.name) is True else "false" if getattr(s, f.name) is False else getattr(s, f.name))
                for f in fields(s)}

    @classmethod
    def from_mapping(cls, mapping) -> "OringSettings":
        d = cls()
        raw = {f.name: mapping.get(f.name, getattr(d, f.name)) for f in fields(cls)}
        return cls(**raw).clean()


def load_settings(path: str = ini_store.CONFIG_PATH) -> OringSettings:
    parser = ini_store.read_ini(path)
    if parser.has_section(SECTION):
        return OringSettings.from_mapping(dict(parser[SECTION]))
    return OringSettings()


def save_settings(settings: OringSettings, path: str = ini_store.CONFIG_PATH, backup: bool = True) -> OringSettings:
    s = settings.clean()
    ini_store.update_ini(SECTION, s.to_section(), path, backup=backup)
    return s
