import pytest

from receval import cli, config


def test_help_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0
    assert "receval" in capsys.readouterr().out


def test_windows_do_not_overlap_the_test_period():
    assert config.TRAIN[1] < config.VALID[0] <= config.VALID[1] < config.TEST[0]
    assert config.CALIB[1] < config.TEST[0]
