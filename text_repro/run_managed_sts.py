import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime


OOM_PATTERNS = [
    "CUDA out of memory",
    "OutOfMemoryError",
    "CUBLAS_STATUS_ALLOC_FAILED",
    "CUDA error: out of memory",
]


def timestamp():
    return datetime.now().isoformat(
        timespec="seconds"
    )


def remove_if_exists(path):
    path = Path(path)
    if path.exists():
        path.unlink()


def write_marker(path):
    Path(path).touch()


def get_gpu_snapshot():
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        return result.stdout
    except Exception as e:
        return f"Could not run nvidia-smi: {e}\n"


def running_process_is_alive(run_dir):
    status_path = run_dir / "status.json"

    if not status_path.exists():
        return False

    try:
        with open(status_path) as f:
            status = json.load(f)

        pid = status.get("pid")

        if pid is None:
            return False

        proc_path = Path(
            f"/proc/{pid}/cmdline"
        )

        if not proc_path.exists():
            return False

        cmdline = proc_path.read_bytes().replace(
            b"\x00",
            b" ",
        ).decode(
            errors="ignore"
        )

        return "train_sts.py" in cmdline

    except Exception:
        return False


def classify_failure(log_text):
    for pattern in OOM_PATTERNS:
        if pattern.lower() in log_text.lower():
            return "FAILED_OOM"

    return "FAILED_OTHER"


def run_one(args, seed):
    output_root = Path(args.output_dir)

    experiment_dir = (
        output_root
        / "experiments"
        / args.experiment_name
    )

    run_dir = (
        experiment_dir
        / f"seed_{seed}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    success_marker = (
        run_dir / "SUCCESS"
    )

    running_marker = (
        run_dir / "RUNNING"
    )

    oom_marker = (
        run_dir / "FAILED_OOM"
    )

    other_marker = (
        run_dir / "FAILED_OTHER"
    )

    if success_marker.exists():
        print(
            f"[seed {seed}] SUCCESS already exists, skip."
        )
        return

    if running_marker.exists():
        if running_process_is_alive(run_dir):
            print(
                f"[seed {seed}] still RUNNING, skip."
            )
            return
        else:
            print(
                f"[seed {seed}] stale RUNNING marker found; rerunning."
            )
            running_marker.unlink()

    remove_if_exists(oom_marker)
    remove_if_exists(other_marker)

    write_marker(running_marker)

    gpu_before = get_gpu_snapshot()

    with open(
        run_dir / "gpu_before.txt",
        "w",
    ) as f:
        f.write(gpu_before)

    cmd = [
        sys.executable,
        "text_repro/train_sts.py",
        "--pooling",
        args.pooling,
        "--seed",
        str(seed),
        "--experiment_name",
        args.experiment_name,
        "--output_dir",
        args.output_dir,
    ]

    if args.pooling in {
        "STFT_FLaG",
        "STFT_FLaG_Pos",
    }:
        cmd.extend([
            "--stft_win_length",
            str(args.stft_win_length),
            "--stft_hop_length",
            str(args.stft_hop_length),
            "--stft_window_type",
            args.stft_window_type,
        ])
        if args.stft_center:
            cmd.append(
                "--stft_center"
            )

    if args.fixed_fft_length is not None:
        if args.pooling not in {
            "FLaG",
            "FLaG_Hann",
        }:
            raise ValueError(
                "--fixed_fft_length is only valid for "
                "global FLaG/FLaG_Hann."
            )

        cmd.extend([
            "--fixed_fft_length",
            str(args.fixed_fft_length),
        ])

    print()
    print("=" * 70)
    print(
        f"Starting {args.experiment_name}, seed {seed}"
    )
    print("=" * 70)
    print(" ".join(cmd))
    print()

    log_path = (
        run_dir / "train.log"
    )

    with open(
        log_path,
        "w",
        buffering=1,
    ) as log_file:

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        status = {
            "experiment": args.experiment_name,
            "pooling": args.pooling,
            "seed": seed,
            "status": "RUNNING",
            "start_time": timestamp(),
            "pid": process.pid,
            "command": cmd,
        }

        with open(
            run_dir / "status.json",
            "w",
        ) as f:
            json.dump(
                status,
                f,
                indent=2,
            )

        captured = []

        for line in process.stdout:
            print(
                line,
                end="",
            )

            log_file.write(line)
            captured.append(line)

        return_code = process.wait()

    log_text = "".join(captured)

    remove_if_exists(running_marker)

    gpu_after = get_gpu_snapshot()

    with open(
        run_dir / "gpu_after.txt",
        "w",
    ) as f:
        f.write(gpu_after)

    metrics_path = (
        run_dir / "metrics.json"
    )

    if (
        return_code == 0
        and metrics_path.exists()
    ):
        final_status = "SUCCESS"
        write_marker(success_marker)

    else:
        final_status = classify_failure(
            log_text
        )

        if final_status == "FAILED_OOM":
            write_marker(oom_marker)
        else:
            write_marker(other_marker)

    status.update({
        "status": final_status,
        "end_time": timestamp(),
        "return_code": return_code,
    })

    with open(
        run_dir / "status.json",
        "w",
    ) as f:
        json.dump(
            status,
            f,
            indent=2,
        )

    print()
    print(
        f"[seed {seed}] {final_status}"
    )


def save_experiment_snapshot(args):
    output_root = Path(args.output_dir)

    experiment_dir = (
        output_root
        / "experiments"
        / args.experiment_name
    )

    snapshot_dir = (
        experiment_dir
        / "code_snapshot"
    )

    snapshot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = {
        "train_sts.py":
            Path("text_repro/train_sts.py"),
        "flag_pooling.py":
            Path("factory/pooling/flag_pooling.py"),
        "run_managed_sts.py":
            Path("text_repro/run_managed_sts.py"),
    }

    for dst_name, src in files.items():
        if src.exists():
            shutil.copy2(
                src,
                snapshot_dir / dst_name,
            )

    try:
        git_status = subprocess.run(
            ["git", "status", "--short"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        ).stdout

        with open(
            snapshot_dir / "git_status.txt",
            "w",
        ) as f:
            f.write(git_status)

        git_diff = subprocess.run(
            ["git", "diff"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        ).stdout

        with open(
            snapshot_dir / "git_diff.patch",
            "w",
        ) as f:
            f.write(git_diff)

    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment_name",
        required=True,
    )

    parser.add_argument(
        "--pooling",
        required=True,
        choices=[
            "mean",
            "FLaG",
            "FLaG_Hann",
            "STFT_FLaG",
            "STFT_FLaG_Pos",
        ],
    )

    parser.add_argument(
        "--stft_win_length",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--stft_hop_length",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        default=(
            "/home/data/home/wwr_lumos/"
            "AMPCliff/outputs/text/stsbenchmark"
        ),
    )

    parser.add_argument(
        "--stft_window_type",
        choices=["rect", "hann"],
        default="rect",
    )

    parser.add_argument(
        "--stft_center",
        action="store_true",
    )

    parser.add_argument(
        "--fixed_fft_length",
        type=int,
        default=None,
        help=(
            "Fixed global FFT length passed to train_sts.py."
        ),
    )

    args = parser.parse_args()

    save_experiment_snapshot(args)

    for seed in args.seeds:
        run_one(
            args,
            seed,
        )


if __name__ == "__main__":
    main()
