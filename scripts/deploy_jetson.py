#!/usr/bin/env python3
"""部署到 Jetson（密码认证）。

两种模式：
  全量：打包 tar.gz -> 断点续传上传 -> 远端解压 + 建 venv + 装核心依赖
        python scripts/deploy_jetson.py
  增量：只同步若干文件（改完代码快速推送，不重建环境）
        python scripts/deploy_jetson.py --files main.py agent/agent.py
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REMOTE = "~/Desktop/ZhuoDaoMedAssistant"
REMOTE_TAR = "/tmp/ZhuoDaoMedAssistant-deploy.tar.gz"

_SKIP_PARTS = {".venv", "venv", ".git", "__pycache__", ".pytest_cache"}
_SKIP_SUFFIXES = {".pyc", ".pyo", ".tar.gz"}


def should_pack(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if any(part in _SKIP_PARTS for part in Path(rel).parts):
        return False
    if rel.startswith(("data/recordings/", "data/logs/")):
        return False
    if path.suffix in _SKIP_SUFFIXES or path.name == "deploy.log":
        return False
    return True


def build_tarball(tar_path: Path) -> int:
    count = 0
    with tarfile.open(tar_path, "w:gz") as tar:
        for dirpath, dirnames, filenames in os.walk(ROOT):
            dirnames[:] = [
                d for d in dirnames if should_pack(Path(dirpath) / d) and d not in _SKIP_PARTS
            ]
            for name in filenames:
                local = Path(dirpath) / name
                if not should_pack(local):
                    continue
                tar.add(local, arcname=local.relative_to(ROOT).as_posix())
                count += 1
                if count % 100 == 0:
                    print(f"  packed {count} files...")
    print(f"Archive: {tar_path} ({tar_path.stat().st_size / 1024 / 1024:.1f} MB, {count} files)")
    return count


def connect(host: str, user: str, password: str) -> paramiko.SSHClient:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=user, password=password, timeout=30)
    return ssh


def resolve_remote_dir(ssh: paramiko.SSHClient, remote: str) -> str:
    stdin, stdout, stderr = ssh.exec_command("echo $HOME")
    home = stdout.read().decode().strip()
    return remote.replace("~/", f"{home}/")


def upload_resumable(
    local: Path, host: str, user: str, password: str, remote_tar: str, retries: int = 5
) -> paramiko.SSHClient:
    """断点续传上传（弱网下被断开后从已传字节处续传）。"""
    total = local.stat().st_size
    chunk = 4 * 1024 * 1024
    last_err: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            print(f"Connect attempt {attempt}/{retries}...")
            ssh = connect(host, user, password)
            sftp = ssh.open_sftp()
            try:
                offset = sftp.stat(remote_tar).st_size
            except OSError:
                offset = 0

            if offset >= total:
                print(f"Remote already complete ({offset / 1024 / 1024:.0f} MB)")
                sftp.close()
                return ssh
            if offset:
                print(f"Resuming from {offset / 1024 / 1024:.0f}/{total / 1024 / 1024:.0f} MB")

            last_report = offset
            with local.open("rb") as lf, sftp.open(remote_tar, "ab" if offset else "wb") as rf:
                lf.seek(offset)
                while offset < total:
                    data = lf.read(chunk)
                    if not data:
                        break
                    rf.write(data)
                    offset += len(data)
                    if offset - last_report >= 50 * 1024 * 1024 or offset == total:
                        print(f"  upload {offset / 1024 / 1024:.0f}/{total / 1024 / 1024:.0f} MB")
                        last_report = offset
            sftp.close()
            return ssh
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"  failed: {e}")
    raise SystemExit(f"Upload failed after {retries} attempts: {last_err}")


def run_remote(ssh: paramiko.SSHClient, cmd: str, timeout: int = 900) -> None:
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    code = stdout.channel.recv_exit_status()
    if out.strip():
        print(out)
    if err.strip():
        print(err, file=sys.stderr)
    if code != 0:
        raise SystemExit(f"Remote command failed (exit {code})")


def deploy_full(args: argparse.Namespace) -> None:
    tar_path = ROOT / "ZhuoDaoMedAssistant-deploy.tar.gz"
    if args.skip_pack and tar_path.exists():
        print(f"Reusing {tar_path} ({tar_path.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        print("Packing project...")
        build_tarball(tar_path)

    ssh = upload_resumable(tar_path, args.host, args.user, args.password, REMOTE_TAR)
    remote_dir = resolve_remote_dir(ssh, args.remote_dir)

    print("Extracting and setting up on Jetson...")
    run_remote(
        ssh,
        f"""set -e
mkdir -p {remote_dir}
tar -xzf {REMOTE_TAR} -C {remote_dir}
rm -f {REMOTE_TAR}
cd {remote_dir}
python3 -m venv .venv
.venv/bin/pip install -U pip -q
.venv/bin/pip install pyyaml pydantic jsonschema openai pytest -q
mkdir -p data/patients data/schedules data/dialogues data/recordings
.venv/bin/python -c "from main import load_config; print('config ok')"
echo DEPLOY_OK
""",
    )
    ssh.close()
    if not args.keep_tar:
        tar_path.unlink(missing_ok=True)

    print(f"\nDone: {args.user}@{args.host}:{remote_dir}")
    print(f"On Jetson run:\n  cd {remote_dir} && .venv/bin/python main.py --text")


def deploy_files(args: argparse.Namespace) -> None:
    ssh = connect(args.host, args.user, args.password)
    remote_dir = resolve_remote_dir(ssh, args.remote_dir)
    sftp = ssh.open_sftp()
    for rel in args.files:
        local = (ROOT / rel).resolve()
        if not local.is_file():
            print(f"skip (not a file): {rel}", file=sys.stderr)
            continue
        remote = f"{remote_dir}/{local.relative_to(ROOT).as_posix()}"
        try:
            sftp.stat(os.path.dirname(remote))
        except OSError:
            run_remote(ssh, f"mkdir -p {os.path.dirname(remote)}")
        sftp.put(str(local), remote)
        print(f"uploaded {rel}")
    sftp.close()
    ssh.close()
    print("done")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy ZhuoDaoMedAssistant to Jetson")
    parser.add_argument("--host", default="192.168.2.158")
    parser.add_argument("--user", default="orin")
    parser.add_argument("--password", default="orin")
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE)
    parser.add_argument("--files", nargs="+", help="增量模式：只同步这些文件（相对项目根）")
    parser.add_argument("--keep-tar", action="store_true", help="保留本地 tar.gz")
    parser.add_argument("--skip-pack", action="store_true", help="复用已存在的 tar.gz")
    args = parser.parse_args()

    if args.files:
        deploy_files(args)
    else:
        deploy_full(args)


if __name__ == "__main__":
    main()
