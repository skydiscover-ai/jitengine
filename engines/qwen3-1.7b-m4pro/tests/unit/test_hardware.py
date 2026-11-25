from jitengine_qwen3_1_7b_m4pro import hardware

BATT = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t73%; charging; 1:43 remaining present: true\n"


def test_pmset_battery():
    assert hardware.parse_pmset_batt(BATT) == {"source": "AC Power", "battery_percent": 73}
    assert (
        hardware.parse_pmset_batt("Now drawing from 'Battery Power'")["source"] == "Battery Power"
    )
    assert hardware.parse_pmset_batt("") == {"source": None, "battery_percent": None}


def test_pmset_power_mode():
    assert (
        hardware.parse_pmset_powermode("System-wide power settings:\n powermode            2\n")
        == 2
    )
    assert hardware.parse_pmset_powermode("nothing here") is None


def test_tested_status_is_only_the_measured_mac():
    assert hardware.tested_status("Apple M4 Pro", 24.0) == "tested"
    assert hardware.tested_status("Apple M4 Pro", 48.0) == "untested"
    assert hardware.tested_status("Apple M2", 24.0) == "untested"
    assert hardware.tested_status(None, None) == "untested"
