"""Reuse the identical v2 body-contact and neighbour-motion audit."""
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('previous_motion_audit',HERE.parent/'a025_curve_frame_wind_20261020/audit.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
module.CFG=json.loads((HERE/'config.json').read_text())
if __name__=='__main__':module.main()
