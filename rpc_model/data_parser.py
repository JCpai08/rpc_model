"""Parsers for NAD raw data files (non-JSON block format).

Supported files
---------------
* DX_ZY3_NAD_gps.txt         (key=value header + gpsData_xx blocks)
* DX_ZY3_NAD_att.txt         (key=value header + attData_xx blocks)
* DX_ZY3_NAD_imagingTime.txt (tab/space separated table)
* NAD.cbr                    (first line count + 3-column rows)
* NAD.txt                    (simple key=value lines)
* example_rpc.txt            (RPC text format with units)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


_FLOAT_RE = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


@dataclass
class OrbitSample:
    time_code: float
    date_time: str
    px: float
    py: float
    pz: float
    vx: float
    vy: float
    vz: float


@dataclass
class OrbitData:
    coordinate_type: str
    data_type: str
    group_number: int
    samples: List[OrbitSample]

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        times = np.array([s.time_code for s in self.samples], dtype=float)
        positions = np.array([[s.px, s.py, s.pz] for s in self.samples], dtype=float)
        velocities = np.array([[s.vx, s.vy, s.vz] for s in self.samples], dtype=float)
        return times, positions, velocities


@dataclass
class AttitudeSample:
    time_code: float
    date_time: str
    eulor1: float
    eulor2: float
    eulor3: float
    roll_velocity: float
    pitch_velocity: float
    yaw_velocity: float
    q1: float
    q2: float
    q3: float
    q4: float


@dataclass
class AttitudeData:
    att_roll_fixed_error: float
    att_pitch_fixed_error: float
    att_yaw_fixed_error: float
    att_mode: int
    group_number: int
    samples: List[AttitudeSample]

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        times = np.array([s.time_code for s in self.samples], dtype=float)
        quaternions = np.array([[s.q1, s.q2, s.q3, s.q4] for s in self.samples], dtype=float)
        return times, quaternions


@dataclass
class ImagingTimeData:
    rel_lines: np.ndarray
    times: np.ndarray
    delta_times: np.ndarray


@dataclass
class CBRData:
    declared_count: int
    column_indices: np.ndarray
    angle_1: np.ndarray
    angle_2: np.ndarray


@dataclass
class NADBiasData:
    starttime: float
    pitch: float
    vpitch: float
    roll: float
    vroll: float
    yaw: float
    vyaw: float


@dataclass
class RPCTextData:
    scalar: Dict[str, float]
    line_num_coeff: np.ndarray
    line_den_coeff: np.ndarray
    samp_num_coeff: np.ndarray
    samp_den_coeff: np.ndarray


@dataclass
class NADFileConfig:
    data_dir: str | None = None
    gps: str = "DX_ZY3_NAD_gps.txt"
    attitude: str = "DX_ZY3_NAD_att.txt"
    imaging_time: str = "DX_ZY3_NAD_imagingTime.txt"
    cbr: str = "NAD.cbr"
    nad_txt: str = "NAD.txt"
    example_rpc: str = "example_rpc.txt"

    @classmethod
    def from_json(cls, config_path: str | Path) -> "NADFileConfig":
        path = Path(config_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        valid_keys = set(cls.__annotations__.keys())
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def resolve(self, data_dir: str | Path | None = None) -> Dict[str, Path]:
        if data_dir is None:
            data_dir = self.data_dir if self.data_dir is not None else "."
        data_dir = Path(data_dir)

        def _resolve(p: str) -> Path:
            p_obj = Path(p)
            if p_obj.is_absolute():
                return p_obj
            return data_dir / p_obj

        return {
            "gps": _resolve(self.gps),
            "attitude": _resolve(self.attitude),
            "imaging_time": _resolve(self.imaging_time),
            "cbr": _resolve(self.cbr),
            "nad_txt": _resolve(self.nad_txt),
            "example_rpc": _resolve(self.example_rpc),
        }


def _clean_lines(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [line.strip() for line in text.splitlines() if line.strip()]


def _parse_value(raw: str):
    val = raw.strip().rstrip(";").strip()
    if val.startswith('"') and val.endswith('"'):
        return val[1:-1]
    try:
        if re.fullmatch(r"[+-]?\d+", val):
            return int(val)
        return float(val)
    except ValueError:
        return val


def _parse_block_style_file(path: Path) -> Tuple[Dict[str, object], List[Tuple[str, Dict[str, object]]]]:
    text = path.read_text(encoding="utf-8", errors="ignore")

    blocks: List[Tuple[str, Dict[str, object]]] = []
    block_pattern = re.compile(r"(?P<name>[A-Za-z0-9_]+)\s*=\s*\{(?P<body>.*?)\}", re.DOTALL)
    for match in block_pattern.finditer(text):
        name = match.group("name")
        body = match.group("body")
        fields: Dict[str, object] = {}
        for f in re.finditer(r"([A-Za-z0-9_]+)\s*=\s*(.*?);", body, re.DOTALL):
            key = f.group(1)
            value = _parse_value(f.group(2))
            fields[key] = value
        blocks.append((name, fields))

    text_no_blocks = block_pattern.sub("", text)
    header: Dict[str, object] = {}
    for m in re.finditer(r"([A-Za-z0-9_]+)\s*=\s*(.*?);", text_no_blocks, re.DOTALL):
        header[m.group(1)] = _parse_value(m.group(2))

    return header, blocks


class NADDataParser:
    """Parser collection for NAD raw data files."""

    @staticmethod
    def parse_gps(path: str | Path) -> OrbitData:
        header, blocks = _parse_block_style_file(Path(path))
        samples: List[OrbitSample] = []
        for _, data in blocks:
            samples.append(
                OrbitSample(
                    time_code=float(data["timeCode"]),
                    date_time=str(data["dateTime"]),
                    px=float(data["PX"]),
                    py=float(data["PY"]),
                    pz=float(data["PZ"]),
                    vx=float(data["VX"]),
                    vy=float(data["VY"]),
                    vz=float(data["VZ"]),
                )
            )

        return OrbitData(
            coordinate_type=str(header.get("coordinateType", "")),
            data_type=str(header.get("dataType", "")),
            group_number=int(header.get("groupNumber", len(samples))),
            samples=samples,
        )

    @staticmethod
    def parse_attitude(path: str | Path) -> AttitudeData:
        header, blocks = _parse_block_style_file(Path(path))
        samples: List[AttitudeSample] = []
        for _, data in blocks:
            samples.append(
                AttitudeSample(
                    time_code=float(data["timeCode"]),
                    date_time=str(data["dateTime"]),
                    eulor1=float(data["eulor1"]),
                    eulor2=float(data["eulor2"]),
                    eulor3=float(data["eulor3"]),
                    roll_velocity=float(data["roll_velocity"]),
                    pitch_velocity=float(data["pitch_velocity"]),
                    yaw_velocity=float(data["yaw_velocity"]),
                    q1=float(data["q1"]),
                    q2=float(data["q2"]),
                    q3=float(data["q3"]),
                    q4=float(data["q4"]),
                )
            )

        return AttitudeData(
            att_roll_fixed_error=float(header.get("att_roll_fixed_error", 0.0)),
            att_pitch_fixed_error=float(header.get("att_pitch_fixed_error", 0.0)),
            att_yaw_fixed_error=float(header.get("att_yaw_fixed_error", 0.0)),
            att_mode=int(header.get("AttMode", 0)),
            group_number=int(header.get("groupNumber", len(samples))),
            samples=samples,
        )

    @staticmethod
    def parse_imaging_time(path: str | Path) -> ImagingTimeData:
        lines = _clean_lines(Path(path))
        rows: List[int] = []
        times: List[float] = []
        delta: List[float] = []

        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 3:
                continue
            rows.append(int(parts[0]))
            times.append(float(parts[1]))
            delta.append(float(parts[2]))

        return ImagingTimeData(
            rel_lines=np.array(rows, dtype=int),
            times=np.array(times, dtype=float),
            delta_times=np.array(delta, dtype=float),
        )

    @staticmethod
    def parse_cbr(path: str | Path) -> CBRData:
        lines = _clean_lines(Path(path))
        declared_count = int(lines[0])

        indices: List[int] = []
        angle_1: List[float] = []
        angle_2: List[float] = []

        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 3:
                continue
            indices.append(int(parts[0]))
            angle_1.append(float(parts[1]))
            angle_2.append(float(parts[2]))

        return CBRData(
            declared_count=declared_count,
            column_indices=np.array(indices, dtype=int),
            angle_1=np.array(angle_1, dtype=float),
            angle_2=np.array(angle_2, dtype=float),
        )

    @staticmethod
    def parse_nad_txt(path: str | Path) -> NADBiasData:
        lines = _clean_lines(Path(path))
        mapping: Dict[str, float] = {}
        for line in lines:
            if "=" not in line:
                continue
            key, value = [x.strip() for x in line.split("=", 1)]
            mapping[key] = float(value)

        return NADBiasData(
            starttime=float(mapping.get("starttime", 0.0)),
            pitch=float(mapping.get("pitch", 0.0)),
            vpitch=float(mapping.get("Vpitch", 0.0)),
            roll=float(mapping.get("roll", 0.0)),
            vroll=float(mapping.get("Vroll", 0.0)),
            yaw=float(mapping.get("yaw", 0.0)),
            vyaw=float(mapping.get("Vyaw", 0.0)),
        )

    @staticmethod
    def parse_example_rpc(path: str | Path) -> RPCTextData:
        lines = _clean_lines(Path(path))
        scalar: Dict[str, float] = {}
        line_num = np.zeros(20, dtype=float)
        line_den = np.zeros(20, dtype=float)
        samp_num = np.zeros(20, dtype=float)
        samp_den = np.zeros(20, dtype=float)

        for line in lines:
            if ":" not in line:
                continue
            key, raw = [x.strip() for x in line.split(":", 1)]
            match = _FLOAT_RE.search(raw)
            if match is None:
                continue
            value = float(match.group())

            if key.startswith("LINE_NUM_COEFF_"):
                idx = int(key.split("_")[-1]) - 1
                line_num[idx] = value
            elif key.startswith("LINE_DEN_COEFF_"):
                idx = int(key.split("_")[-1]) - 1
                line_den[idx] = value
            elif key.startswith("SAMP_NUM_COEFF_"):
                idx = int(key.split("_")[-1]) - 1
                samp_num[idx] = value
            elif key.startswith("SAMP_DEN_COEFF_"):
                idx = int(key.split("_")[-1]) - 1
                samp_den[idx] = value
            else:
                scalar[key] = value

        return RPCTextData(
            scalar=scalar,
            line_num_coeff=line_num,
            line_den_coeff=line_den,
            samp_num_coeff=samp_num,
            samp_den_coeff=samp_den,
        )


def load_nad_bundle(
    data_dir: str | Path | None = None,
    config: NADFileConfig | None = None,
    config_path: str | Path | None = None,
) -> Dict[str, object]:
    """Load all NAD files in one call.

    Parameters
    ----------
    data_dir : str | Path, optional
        Base directory for relative paths in config.
        If None, uses ``config.data_dir`` then current directory.
    config : NADFileConfig, optional
        Explicit file mapping object.
    config_path : str | Path, optional
        JSON config file path. Ignored when ``config`` is provided.

    Returns
    -------
    dict
        Keys: orbit, attitude, imaging_time, cbr, nad_txt, example_rpc.
    """
    if config is None:
        config = NADFileConfig.from_json(config_path) if config_path else NADFileConfig()

    paths = config.resolve(data_dir)
    return {
        "orbit": NADDataParser.parse_gps(paths["gps"]),
        "attitude": NADDataParser.parse_attitude(paths["attitude"]),
        "imaging_time": NADDataParser.parse_imaging_time(paths["imaging_time"]),
        "cbr": NADDataParser.parse_cbr(paths["cbr"]),
        "nad_txt": NADDataParser.parse_nad_txt(paths["nad_txt"]),
        "example_rpc": NADDataParser.parse_example_rpc(paths["example_rpc"]),
    }
