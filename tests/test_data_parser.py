"""Tests for NAD raw data parser."""

import json
from pathlib import Path

import numpy as np

from rpc_model.data_parser import NADDataParser, load_nad_bundle


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = WORKSPACE_ROOT / "data" / "config.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
DATA_DIR = WORKSPACE_ROOT / CONFIG["data_dir"]


def data_file(key: str) -> Path:
    return DATA_DIR / CONFIG[key]


class TestNADRawParsers:
    def test_parse_gps(self):
        gps = NADDataParser.parse_gps(data_file("gps"))
        assert gps.group_number == 101
        assert len(gps.samples) == 101
        t, p, v = gps.to_arrays()
        assert t.shape == (101,)
        assert p.shape == (101, 3)
        assert v.shape == (101, 3)
        assert np.all(np.diff(t) > 0.0)

    def test_parse_attitude(self):
        att = NADDataParser.parse_attitude(data_file("attitude"))
        assert att.group_number == 401
        assert len(att.samples) == 401
        t, q = att.to_arrays()
        assert t.shape == (401,)
        assert q.shape == (401, 4)
        norms = np.linalg.norm(q, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)

    def test_parse_imaging_time(self):
        imt = NADDataParser.parse_imaging_time(data_file("imaging_time"))
        assert len(imt.rel_lines) == 5378
        assert len(imt.times) == 5378
        assert len(imt.delta_times) == 5378
        assert imt.rel_lines[0] == 0
        assert imt.rel_lines[-1] == 5377
        assert np.all(np.diff(imt.times) > 0.0)

    def test_parse_cbr(self):
        cbr = NADDataParser.parse_cbr(data_file("cbr"))
        assert cbr.declared_count == 8192
        assert len(cbr.column_indices) == 8192
        assert len(cbr.angle_1) == 8192
        assert len(cbr.angle_2) == 8192
        assert cbr.column_indices[0] == 0
        assert cbr.column_indices[-1] == 8191

    def test_parse_nad_txt(self):
        nad = NADDataParser.parse_nad_txt(data_file("nad_txt"))
        assert nad.starttime == 0.0
        assert nad.pitch == 0.0
        assert nad.roll == 0.0
        assert nad.yaw == 0.0

    def test_parse_example_rpc(self):
        rpc = NADDataParser.parse_example_rpc(data_file("example_rpc"))
        assert "LINE_OFF" in rpc.scalar
        assert "SAMP_OFF" in rpc.scalar
        assert rpc.line_num_coeff.shape == (20,)
        assert rpc.line_den_coeff.shape == (20,)
        assert rpc.samp_num_coeff.shape == (20,)
        assert rpc.samp_den_coeff.shape == (20,)
        np.testing.assert_allclose(rpc.line_den_coeff[0], 1.0, atol=1e-12)
        np.testing.assert_allclose(rpc.samp_den_coeff[0], 1.0, atol=1e-12)

    def test_load_bundle(self):
        bundle = load_nad_bundle(config_path=CONFIG_PATH)
        expected_keys = {"orbit", "attitude", "imaging_time", "cbr", "nad_txt", "example_rpc"}
        assert set(bundle.keys()) == expected_keys

    def test_load_bundle_with_json_config(self, tmp_path):
        cfg_path = tmp_path / "config.json"
        config_copy = dict(CONFIG)
        config_copy["data_dir"] = str(DATA_DIR)
        cfg_path.write_text(
            json.dumps(config_copy, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        bundle = load_nad_bundle(config_path=cfg_path)
        expected_keys = {"orbit", "attitude", "imaging_time", "cbr", "nad_txt", "example_rpc"}
        assert set(bundle.keys()) == expected_keys
