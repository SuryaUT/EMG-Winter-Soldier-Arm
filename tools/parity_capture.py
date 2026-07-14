"""
Capture a PARITY_DUMP run off the ESP32 into a clean log file.

Reads the serial port until the device prints #PARITY_END, then writes the raw
lines to disk. Use this instead of piping `pio device monitor`, which injects
terminal escape codes that corrupt the numeric parse.

Notes for the ESP32-S3's native USB-Serial-JTAG (VID 303A):
  * DTR/RTS are NOT flow-control lines there — they drive reset/boot mode. We
    explicitly de-assert them, otherwise the chip can sit in reset or drop into
    download mode and emit nothing.
  * Resetting the board re-enumerates the USB device, which invalidates the open
    handle ("ClearCommError failed"). We reopen and keep going instead of dying.

Usage:
    python tools/parity_capture.py --out dump.log
    python tools/parity_capture.py --port COM6 --out dump.log
"""
import argparse
import sys
import time

import serial
from serial.tools import list_ports

ESPRESSIF_VID = 0x303A

ap = argparse.ArgumentParser()
ap.add_argument("--port", default=None, help="e.g. COM6 (auto-detects Espressif VID if omitted)")
ap.add_argument("--baud", type=int, default=921600, help="ignored by USB-CDC, matters for UART bridges")
ap.add_argument("--out", default="dump.log")
ap.add_argument("--timeout", type=float, default=240.0, help="seconds to wait for #PARITY_END")
args = ap.parse_args()


def pick_port():
    ports = list(list_ports.comports())
    if not ports:
        sys.exit("No serial ports found. Is the ESP32 plugged in?")
    esp = [p for p in ports if (p.vid or 0) == ESPRESSIF_VID]
    chosen = (esp or ports)[0]
    print(f"Port: {chosen.device}  ({chosen.description})")
    return chosen.device


def open_port(port):
    """Open without touching DTR/RTS — see module docstring."""
    ser = serial.Serial()
    ser.port = port
    ser.baudrate = args.baud
    ser.timeout = 1.0
    ser.rtscts = False
    ser.dsrdtr = False
    ser.dtr = False
    ser.rts = False
    ser.open()
    try:
        ser.reset_input_buffer()
    except Exception:
        pass
    return ser


port = args.port or pick_port()
print("Waiting for the dump. If nothing arrives in ~10s, tap the RESET/EN button.\n")

lines, saw_begin, done = [], False, False
t0 = time.time()
ser = None
try:
    while time.time() - t0 < args.timeout and not done:
        try:
            if ser is None:
                ser = open_port(port)
            raw = ser.readline()
        except (serial.SerialException, OSError) as e:
            # Board reset -> USB re-enumeration. Drop the stale handle and retry.
            if ser is not None:
                try:
                    ser.close()
                except Exception:
                    pass
                ser = None
            print(f"  [reconnecting: {type(e).__name__}]        ", end="\r")
            time.sleep(0.5)
            continue

        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")

        if line.startswith("#PARITY_BEGIN"):
            saw_begin, lines = True, []      # restart cleanly if the board reboots mid-dump
            print("  -> dump started                      ")
            continue
        if not saw_begin:
            continue

        lines.append(line)
        if len(lines) % 50 == 0:
            print(f"  ... {len(lines)} lines", end="\r")
        if line.startswith("#PARITY_END"):
            print(f"\n  -> dump complete: {line}")
            done = True
finally:
    if ser is not None:
        try:
            ser.close()
        except Exception:
            pass

if not lines:
    sys.exit("Nothing captured. Confirm MAIN_MODE is PARITY_DUMP, then reset the board.")
if not done:
    print("\n[warn] never saw #PARITY_END — saving the partial capture anyway.")

with open(args.out, "w") as f:
    f.write("\n".join(lines) + "\n")
print(f"Wrote {len(lines)} lines to {args.out}")
print(f"\nNow run:\n    python tools/parity_compare.py {args.out}")
