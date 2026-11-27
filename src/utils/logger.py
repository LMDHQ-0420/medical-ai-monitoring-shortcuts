"""按实验保存最近一次运行日志；同时保留终端输出。"""
from contextlib import ExitStack, contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import fcntl
import hashlib
from pathlib import Path
import shlex
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]


class _Tee:
    def __init__(self, terminal, files):
        self.terminal, self.files = terminal, files

    def write(self, text):
        self.terminal.write(text)
        for file in self.files:
            file.write(text)
            file.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        for file in self.files:
            file.flush()


def code_identity():
    """记录提交号及实际源文件指纹，未提交修改也能区分。"""
    digest = hashlib.sha256()
    for folder in ('src', 'scripts', 'configs'):
        for path in sorted((ROOT / folder).rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.yaml', '.yml', '.json'):
                digest.update(str(path.relative_to(ROOT)).encode() + b'\0')
                digest.update(path.read_bytes() + b'\0')
    result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                            capture_output=True, text=True)
    return f'commit={result.stdout.strip() or "unknown"} source_sha256={digest.hexdigest()}'


@contextmanager
def experiment_log(experiments, name, *, log_root=None, detail=''):
    """同名日志覆盖；共同生成 exp4/5 的运行在两处留相同记录。"""
    if isinstance(experiments, str):
        experiments = (experiments,)
    if not experiments or any(exp not in ('exp1', 'exp2', 'exp3', 'exp4', 'exp5') for exp in experiments):
        raise ValueError('日志必须归属 exp1–exp5')
    if not name or Path(name).name != name:
        raise ValueError('日志名称不能包含目录')
    base = Path(log_root) if log_root is not None else ROOT / 'logs'
    with ExitStack() as stack:
        files = []
        for exp in sorted(set(experiments)):
            path = base / exp / f'{name}.log'
            path.parent.mkdir(parents=True, exist_ok=True)
            file = stack.enter_context(path.open('a+', encoding='utf-8'))
            # 先锁定再清空，避免并行命令覆盖仍在写入的同名日志。
            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            files.append(file)
        for file in files:
            file.seek(0)
            file.truncate()
        with redirect_stdout(_Tee(sys.stdout, files)), redirect_stderr(_Tee(sys.stderr, files)):
            print(f'START {datetime.now(timezone.utc).isoformat()} experiments={",".join(experiments)}')
            print(code_identity())
            print('command=' + shlex.join([sys.executable, *sys.argv]))
            print(detail)
            try:
                yield
            except BaseException:
                traceback.print_exc()
                print('STATUS failed_or_interrupted')
                raise
            else:
                print('STATUS complete')
            finally:
                print('END ' + datetime.now(timezone.utc).isoformat())
