"""验证实验日志覆盖、错误记录及共享评估归档，不运行正式训练。"""
import sys
import pytest
from src.utils.logger import experiment_log


def test_log_overwrites_and_captures_both_streams(tmp_path):
    with experiment_log('exp1', 'cell', log_root=tmp_path):
        print('previous-run')
    with experiment_log('exp1', 'cell', log_root=tmp_path):
        print('current-run')
        print('warning-message', file=sys.stderr)
    text = (tmp_path / 'exp1/cell.log').read_text()
    assert 'previous-run' not in text
    assert all(s in text for s in ('current-run', 'warning-message', 'STATUS complete', 'source_sha256=', 'command='))


def test_failure_is_recorded_and_streams_restored(tmp_path):
    original = sys.stdout
    with pytest.raises(RuntimeError):
        with experiment_log('exp2', 'match', log_root=tmp_path):
            raise RuntimeError('intentional-test-error')
    text = (tmp_path / 'exp2/match.log').read_text()
    assert 'intentional-test-error' in text and 'STATUS failed_or_interrupted' in text
    assert 'STATUS complete' not in text
    assert sys.stdout is original


def test_shared_evaluation_and_active_log_lock(tmp_path):
    with experiment_log(('exp4', 'exp5'), 'cell', log_root=tmp_path):
        print('shared evaluation')
        with pytest.raises(BlockingIOError):
            with experiment_log('exp4', 'cell', log_root=tmp_path):
                pytest.fail('同名运行日志不能并发覆盖')
    assert (tmp_path / 'exp4/cell.log').read_bytes() == (tmp_path / 'exp5/cell.log').read_bytes()
    assert 'shared evaluation' in (tmp_path / 'exp4/cell.log').read_text()


def test_unknown_experiment_rejected(tmp_path):
    with pytest.raises(ValueError):
        with experiment_log('old_grid', 'cell', log_root=tmp_path):
            pass
