"""
@file serial_stream.py
@brief Real serial stream for reading EMG data from ESP32 with handshake protocol.

This module provides a serial communication interface for receiving
EMG data from the ESP32 microcontroller over USB. It implements a
robust handshake protocol to ensure reliable connection before streaming.

@section usage Usage Example
@code{.py}
    from serial_stream import RealSerialStream

    # Create stream (auto-detects port, or specify manually)
    stream = RealSerialStream(port='COM3')

    # Connect with handshake (raises on timeout/failure)
    stream.connect(timeout=5.0)

    # Start streaming
    stream.start()

    # Read data
    while True:
        line = stream.readline()
        if line:
            print(line)  # "1234,512,489,501,523"

    # Stop streaming (device returns to CONNECTED state)
    stream.stop()

    # Disconnect (device returns to IDLE state)
    stream.disconnect()
@endcode

@section protocol Handshake Protocol & State Machine

    ESP32 States:
        IDLE      - Waiting for "connect" command
        CONNECTED - Handshake complete, waiting for "start" command
        STREAMING - Actively sending CSV data

    State Transitions:
        1. connect()  : IDLE → CONNECTED
           App sends: {"cmd": "connect"}
           Device responds: {"status": "ack_connect", "device": "ESP32-EMG", "channels": 4}

        2. start()    : CONNECTED → STREAMING
           App sends: {"cmd": "start"}
           Device starts streaming CSV data

        3. stop()     : STREAMING → CONNECTED
           App sends: {"cmd": "stop"}
           Device stops streaming, ready for new start command

        4. disconnect() : ANY → IDLE
           App sends: {"cmd": "disconnect"}
           Device returns to idle, waiting for new connection

@section format Data Format
    The ESP32 sends data as CSV lines:
    "timestamp_ms,ch0,ch1,ch2,ch3\\n"

    Example:
    "12345,512,489,501,523\\n"

@author Bucky Arm Project
"""

import serial
import serial.tools.list_ports
import json
import time
import threading
from typing import Optional, List, Dict, Any
from enum import Enum


class ConnectionState(Enum):
    """Connection states for the serial stream."""
    DISCONNECTED = 0  # No serial connection
    CONNECTING = 1     # Serial open, waiting for handshake
    CONNECTED = 2      # Handshake complete, ready to stream
    STREAMING = 3      # Actively streaming data


class RealSerialStream:
    """
    @brief Reads EMG data from ESP32 over USB serial with handshake protocol.

    This class implements a robust connection protocol:
        - connect()    : Open serial port and perform handshake
        - start()      : Begin streaming data
        - stop()       : Stop streaming (device stays connected)
        - disconnect() : Close connection (device returns to idle)
        - readline()   : Read one line of data

    State machine ensures reliable communication without timing dependencies.

    @note Requires pyserial: pip install pyserial
    """

    def __init__(self, port: str = None, baud_rate: int = 921600, timeout: float = 0.05):
        """
        @brief Initialize the serial stream.

        @param port      Serial port name (e.g., 'COM3' on Windows, '/dev/ttyUSB0' on Linux).
                         If None, will attempt to auto-detect the ESP32.
        @param baud_rate Communication speed in bits per second. Default 921600 for high-throughput streaming.
        @param timeout   Read timeout in seconds for readline().
        """
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.serial: Optional[serial.Serial] = None
        self.state = ConnectionState.DISCONNECTED
        self.device_info: Optional[Dict[str, Any]] = None
        self._auto_detect_result: Optional[Dict[str, Any]] = None  # Cache for concurrent auto-detect
        self._rx_buffer = b''  # Byte buffer for bulk-read line parsing (see _readline_raw)

    def connect(self, timeout: float = 5.0) -> Dict[str, Any]:
        """
        @brief Connect to the ESP32 and perform handshake.

        Opens the serial port, sends connect command, and waits for
        acknowledgment from the device.

        @param timeout Maximum time to wait for handshake response (seconds).

        @return Device info dict from handshake response.

        @throws RuntimeError If no port specified and auto-detect fails.
        @throws RuntimeError If unable to open the serial port.
        @throws TimeoutError If device doesn't respond within timeout.
        @throws ValueError If device sends invalid handshake response.
        """
        if self.state != ConnectionState.DISCONNECTED:
            raise RuntimeError(f"Already in state {self.state.name}, cannot connect")

        # Auto-detect port if not specified
        if self.port is None:
            self.port = self._auto_detect_port()

        if self.port is None:
            raise RuntimeError(
                "No serial port specified and auto-detect failed.\n"
                "Use RealSerialStream.list_ports() to see available ports."
            )

        # Open serial connection
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baud_rate,
                timeout=self.timeout
            )
            self.state = ConnectionState.CONNECTING

            # Clear any stale data in the buffer
            self.serial.reset_input_buffer()
            self._rx_buffer = b''
            time.sleep(0.1)  # Let device settle after port open

            print(f"[SERIAL] Port opened: {self.port}")

        except serial.SerialException as e:
            self.state = ConnectionState.DISCONNECTED
            error_msg = f"Failed to open {self.port}: {e}"
            if "Permission denied" in str(e) or "Resource busy" in str(e):
                error_msg += "\n\nThe port may still be in use. Wait a moment and try again."
            raise RuntimeError(error_msg)

        # Perform handshake
        try:
            # Send connect command (works from any device state - handles reconnection)
            connect_cmd = {"cmd": "connect"}
            self._send_json(connect_cmd)
            print(f"[SERIAL] Sent: {connect_cmd}")

            # Wait for acknowledgment
            start_time = time.time()
            while (time.time() - start_time) < timeout:
                line = self._readline_raw()
                if line:
                    try:
                        response = json.loads(line)
                        if response.get("status") == "ack_connect":
                            self.device_info = response
                            self.state = ConnectionState.CONNECTED
                            print(f"[SERIAL] Handshake complete: {response}")
                            return response
                    except json.JSONDecodeError:
                        # Ignore non-JSON lines (startup messages, residual CSV data from reconnection)
                        # This allows reconnection even if device was streaming when app crashed
                        if line and line[0].isdigit() and ',' in line:
                            # Residual CSV data - ignore silently
                            pass
                        else:
                            # Other non-JSON - show for debugging
                            print(f"[SERIAL] Ignoring: {line.strip()}")
                        continue

            # Timeout reached
            self.state = ConnectionState.DISCONNECTED
            if self.serial:
                self.serial.close()
                self.serial = None
            raise TimeoutError(
                f"Device did not respond to connection request within {timeout}s.\n"
                "Check that the correct firmware is flashed and device is powered on."
            )

        except Exception as e:
            # Clean up on any error
            self.state = ConnectionState.DISCONNECTED
            if self.serial:
                try:
                    self.serial.close()
                except:
                    pass
                self.serial = None
            raise

    def start(self) -> None:
        """
        @brief Start streaming EMG data.

        Device must be in CONNECTED state. Sends start command to ESP32,
        which begins streaming CSV data.

        @throws RuntimeError If not connected.
        """
        if self.state != ConnectionState.CONNECTED:
            raise RuntimeError(
                f"Cannot start streaming from state {self.state.name}. "
                "Must call connect() first."
            )

        # Flush any stale data before starting fresh stream
        self.serial.reset_input_buffer()

        # Send start command
        start_cmd = {"cmd": "start"}
        self._send_json(start_cmd)
        self.state = ConnectionState.STREAMING
        print(f"[SERIAL] Started streaming")

    def start_predict(self) -> None:
        """
        @brief Start on-device prediction (Edge Inference).
        
        Sends 'start_predict' command. Device enters PREDICTING state.
        Stream receives JSON telemetry instead of raw CSV.
        """
        if self.state != ConnectionState.CONNECTED:
             raise RuntimeError(
                f"Cannot start prediction from state {self.state.name}. "
                "Must call connect() first."
            )
        
        self.serial.reset_input_buffer()
        
        cmd = {"cmd": "start_predict"}
        self._send_json(cmd)
        self.state = ConnectionState.STREAMING # Treat as streaming for readline purposes
        print(f"[SERIAL] Started prediction mode")

    def stop(self) -> None:
        """
        @brief Stop streaming EMG data.

        Sends stop command to ESP32, which stops streaming and returns
        to CONNECTED state. Connection remains open for restart.

        Safe to call even if not streaming.
        """
        if self.state == ConnectionState.STREAMING:
            try:
                stop_cmd = {"cmd": "stop"}
                self._send_json(stop_cmd)
                self.state = ConnectionState.CONNECTED
                print(f"[SERIAL] Stopped streaming")
            except Exception as e:
                print(f"[SERIAL] Warning during stop: {e}")

    def disconnect(self) -> None:
        """
        @brief Disconnect from the ESP32.

        Sends disconnect command (device returns to IDLE state),
        then closes the serial port. Safe to call from any state.
        """
        # Send disconnect command if connected
        if self.state in (ConnectionState.CONNECTED, ConnectionState.STREAMING):
            try:
                disconnect_cmd = {"cmd": "disconnect"}
                self._send_json(disconnect_cmd)
                time.sleep(0.1)  # Give device time to process
                print(f"[SERIAL] Sent disconnect command")
            except Exception as e:
                print(f"[SERIAL] Warning sending disconnect: {e}")

        # Close serial port
        if self.serial:
            try:
                if self.serial.is_open:
                    self.serial.close()
                    print(f"[SERIAL] Port closed: {self.port}")
            except Exception as e:
                print(f"[SERIAL] Warning during port close: {e}")
            finally:
                self.serial = None

        self.state = ConnectionState.DISCONNECTED
        self.device_info = None

    def readline(self) -> Optional[str]:
        """
        @brief Read one line of CSV data from the ESP32.

        Should only be called when in STREAMING state.
        Blocks until a complete line is received or timeout occurs.

        @return Line string including newline, or None if timeout/error.

        @note Lines from ESP32 are in format: "timestamp_ms,ch0,ch1,ch2,ch3\\n"
        """
        if self.state != ConnectionState.STREAMING:
            return None

        return self._readline_raw()

    def _readline_raw(self) -> Optional[str]:
        """
        @brief Read one line from serial port (internal helper).

        Drains all bytes currently waiting on the port in a single bulk read
        and pops one complete line from an internal buffer. This replaces
        pyserial's byte-at-a-time ``readline()``, which cannot keep up with the
        1 kHz stream: when the reader falls behind, the OS receive buffer backs
        up and every parsed sample is stamped later than it was captured,
        misaligning training labels. Bulk reads keep the reader caught up to
        real time so receipt timestamps track capture time.

        @return Decoded line string, or None if no complete line is available.
        """
        if not self.serial or not self.serial.is_open:
            return None

        try:
            # If we don't already have a full line buffered, pull whatever is
            # waiting (or block up to `timeout` for at least one byte).
            if b'\n' not in self._rx_buffer:
                waiting = self.serial.in_waiting
                chunk = self.serial.read(waiting if waiting > 0 else 1)
                if not chunk:
                    return None  # timeout, no data
                self._rx_buffer += chunk
                if b'\n' not in self._rx_buffer:
                    return None  # partial line so far; caller will retry

            line, _, self._rx_buffer = self._rx_buffer.partition(b'\n')
            return line.decode('utf-8', errors='ignore').strip()

        except serial.SerialException:
            return None

    def _send_json(self, data: Dict[str, Any]) -> None:
        """
        @brief Send JSON command to the device (internal helper).

        @param data Dictionary to send as JSON.

        @throws RuntimeError If serial port is not open.
        """
        if not self.serial or not self.serial.is_open:
            raise RuntimeError("Serial port not open")

        json_str = json.dumps(data) + "\n"
        self.serial.write(json_str.encode('utf-8'))
        self.serial.flush()  # Ensure data is sent immediately

    def _auto_detect_port(self) -> Optional[str]:
        """
        @brief Attempt to auto-detect the ESP32 serial port using concurrent handshake.

        Sends connect command to all available ports simultaneously and selects
        the first one that responds with valid handshake acknowledgment.

        This is more reliable than USB chip detection since it verifies the
        device is actually running the expected firmware.

        @return Port name if found, None otherwise.

        @note Stores device_info in self._auto_detect_result for reuse by connect()
        """
        ports = serial.tools.list_ports.comports()

        if not ports:
            print("[SERIAL] No serial ports found")
            return None

        if len(ports) == 1:
            # Only one port, no need for concurrent detection
            print(f"[SERIAL] Only one port available: {ports[0].device}")
            return ports[0].device

        print(f"[SERIAL] Auto-detecting ESP32 across {len(ports)} ports...")

        # Thread-safe result container
        result_lock = threading.Lock()
        result = {'port': None, 'device_info': None}

        def try_handshake(port_name: str):
            """Attempt handshake on a single port (runs in thread)."""
            try:
                # Open serial connection with short timeout
                ser = serial.Serial(
                    port=port_name,
                    baudrate=self.baud_rate,
                    timeout=0.5
                )

                # Clear buffer and settle
                ser.reset_input_buffer()
                time.sleep(0.05)

                # Send connect command (resets device state if needed)
                connect_cmd = {"cmd": "connect"}
                json_str = json.dumps(connect_cmd) + "\n"
                ser.write(json_str.encode('utf-8'))
                ser.flush()

                # Wait for acknowledgment (max 2 seconds)
                start_time = time.time()
                while (time.time() - start_time) < 2.0:
                    line_bytes = ser.readline()
                    if line_bytes:
                        try:
                            line = line_bytes.decode('utf-8', errors='ignore').strip()
                            response = json.loads(line)
                            if response.get("status") == "ack_connect":
                                # Found it! Store result if we're first
                                with result_lock:
                                    if result['port'] is None:
                                        result['port'] = port_name
                                        result['device_info'] = response
                                        print(f"[SERIAL] ✓ ESP32 found on {port_name}: {response.get('device', 'Unknown')}")
                                # Send disconnect to return device to IDLE
                                disconnect_cmd = {"cmd": "disconnect"}
                                json_str = json.dumps(disconnect_cmd) + "\n"
                                ser.write(json_str.encode('utf-8'))
                                ser.flush()
                                time.sleep(0.05)
                                ser.close()
                                return
                        except json.JSONDecodeError:
                            # Ignore non-JSON (residual CSV data, startup messages)
                            continue

                # No valid response
                ser.close()

            except (serial.SerialException, OSError):
                # Port unavailable or busy - skip silently
                pass

        # Launch concurrent handshake attempts
        threads = []
        for port in ports:
            thread = threading.Thread(target=try_handshake, args=(port.device,), daemon=True)
            thread.start()
            threads.append(thread)

        # Poll for first success or timeout (instead of sequential joins)
        start_time = time.time()
        max_wait = 2.5

        while (time.time() - start_time) < max_wait:
            # Check if we found a device
            with result_lock:
                if result['port'] is not None:
                    # Success! Return immediately
                    self._auto_detect_result = result
                    elapsed = time.time() - start_time
                    print(f"[SERIAL] Auto-detect complete in {elapsed:.2f}s")
                    return result['port']

            # Check if all threads finished (no device found)
            if not any(thread.is_alive() for thread in threads):
                break

            # Brief sleep to avoid busy-waiting CPU
            time.sleep(0.05)

        # Timeout or all threads finished without success
        print("[SERIAL] No ESP32 responded to handshake on any port")
        return None

    @staticmethod
    def list_ports() -> List[str]:
        """
        @brief List all available serial ports on the system.

        Useful for finding the correct port name for your ESP32.

        @return List of port names (e.g., ['COM3', 'COM4'] on Windows).
        """
        ports = serial.tools.list_ports.comports()

        if not ports:
            print("No serial ports found.")
            return []

        print("\nAvailable serial ports:")
        print("-" * 50)

        for port in ports:
            print(f"  {port.device}")
            print(f"      Description: {port.description}")
            print()

        return [p.device for p in ports]


# =============================================================================
# Standalone Test
# =============================================================================

if __name__ == "__main__":
    """
    @brief Quick test to verify ESP32 serial communication.

    Run this file directly to test:
        python serial_stream.py [port]

    If port is not specified, auto-detection is attempted.
    """
    import sys

    print("=" * 50)
    print("  ESP32 Serial Stream Test")
    print("=" * 50)
    print()

    # Show available ports
    ports = RealSerialStream.list_ports()

    if not ports:
        print("No ports found. Is the ESP32 plugged in?")
        sys.exit(1)

    # Determine which port to use
    if len(sys.argv) > 1:
        port = sys.argv[1]
        print(f"Using specified port: {port}")
    else:
        port = None  # Will auto-detect
        print("No port specified, will auto-detect.")

    print()
    print("Starting stream... (Ctrl+C to stop)")
    print("-" * 50)

    # Create stream
    stream = RealSerialStream(port=port)

    try:
        # Connect with handshake
        print("\nConnecting to device...")
        device_info = stream.connect(timeout=5.0)
        print(f"Connected! Device: {device_info.get('device', 'Unknown')}, "
              f"Channels: {device_info.get('channels', '?')}")
        print()

        # Start streaming
        print("Starting data stream...")
        stream.start()
        print()

        sample_count = 0

        while True:
            line = stream.readline()

            if line:
                # Check if this is a data line (starts with digit = timestamp)
                if line and line[0].isdigit():
                    sample_count += 1

                    # Print every 500th sample to avoid flooding terminal
                    #if sample_count % 500 == 0:
                    print(f"  [{sample_count:6d} samples] Latest: {line}")

                else:
                    # Print non-data messages
                    print(f"  {line}")

    except KeyboardInterrupt:
        print("\n")
        print("-" * 50)
        print("Stopped by user (Ctrl+C)")

    except TimeoutError as e:
        print(f"\nConnection timeout: {e}")

    except Exception as e:
        print(f"\nError: {e}")

    finally:
        print("\nCleaning up...")
        stream.stop()
        stream.disconnect()
        print(f"Total samples received: {sample_count}")
