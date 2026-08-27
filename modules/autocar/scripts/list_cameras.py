"""Probe webcam indices and report which ones are usable (resolution/FPS) -
use a working index as `--source` for main.py. PC-only dev helper, not needed
on the Jetson Nano.

Windows device names (best-effort, via PowerShell) are printed separately
since OpenCV gives no reliable way to map an index to a friendly name - use
them only to tell "which physical camera is which" apart, not to match order.
"""
import platform
import subprocess
import sys

import cv2

MAX_INDEX_TO_PROBE = 10


def probe_camera(index: int):
    # Deliberately not forcing CAP_DSHOW on Windows: it errors out on this
    # setup ("can't be used to capture by index") and gives bogus results
    # (fps=-1, phantom devices). The default backend picks a working one.
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return width, height, fps


def list_windows_device_names():
    if platform.system() != "Windows":
        return []
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_PnPEntity | "
             "Where-Object { $_.PNPClass -eq 'Camera' } | "
             "Select-Object -ExpandProperty Name"],
            capture_output=True, text=True, timeout=10,
        )
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def main():
    names = list_windows_device_names()
    if names:
        print("Windows camera devices (PNPClass=Camera; order not guaranteed to match the index below):")
        for name in names:
            print(f"  - {name}")
        print()

    print(f"Probing camera indices 0-{MAX_INDEX_TO_PROBE - 1}...")
    usable = []
    for i in range(MAX_INDEX_TO_PROBE):
        result = probe_camera(i)
        if result is None:
            print(f"  [{i}] not available")
            continue
        width, height, fps = result
        print(f"  [{i}] OK - {width}x{height} @ {fps:.0f}fps  ->  py main.py --source {i}")
        usable.append(i)

    if not usable:
        print("\nNo usable camera found.")
        sys.exit(1)
    print(f"\nUsable indices: {usable}")


if __name__ == "__main__":
    main()
