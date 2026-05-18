#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import queue
import signal
import subprocess
import sys
import threading
import time


DEFAULT_IDLE_TIMEOUT_SECONDS = 180.0
_EOF = object()


def _reader(pipe, out_queue: "queue.Queue[str | object]") -> None:
    try:
        for line in iter(pipe.readline, ""):
            out_queue.put(line)
    finally:
        out_queue.put(_EOF)


def _kill_process_tree(proc: subprocess.Popen[str]) -> None:
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a command and kill it if stdout/stderr is idle too long."
    )
    parser.add_argument("--idle-timeout", type=float, default=DEFAULT_IDLE_TIMEOUT_SECONDS)
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    if not args.cmd:
        parser.error("missing command")

    proc = subprocess.Popen(
        args.cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        preexec_fn=os.setsid if hasattr(os, "setsid") else None,
    )
    assert proc.stdout is not None
    out_queue: "queue.Queue[str | object]" = queue.Queue()
    thread = threading.Thread(target=_reader, args=(proc.stdout, out_queue), daemon=True)
    thread.start()

    last_output = time.monotonic()
    output_closed = False
    while True:
        try:
            item = out_queue.get(timeout=0.5)
        except queue.Empty:
            item = None
        if item is _EOF:
            output_closed = True
        elif item:
            sys.stdout.write(item)
            sys.stdout.flush()
            last_output = time.monotonic()

        if proc.poll() is not None and output_closed:
            return proc.returncode

        if time.monotonic() - last_output > args.idle_timeout:
            print(
                f"[K1 idle-timeout] no output for {args.idle_timeout:.0f}s; killing: "
                + " ".join(args.cmd),
                flush=True,
            )
            _kill_process_tree(proc)
            proc.wait()
            return 124


if __name__ == "__main__":
    raise SystemExit(main())
