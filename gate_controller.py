import threading
import serial
import serial.tools.list_ports


class GateController:
    def __init__(self, port, baudrate=9600, open_cmd=b'OPEN\n',
                 close_cmd=b'CLOSE\n', open_duration=5.0):
        self.port          = port
        self.baudrate      = baudrate
        self.open_cmd      = open_cmd
        self.close_cmd     = close_cmd
        self.open_duration = open_duration
        self.ser           = None
        self.connected     = False
        self._close_timer  = None
        self._lock         = threading.Lock()

    @staticmethod
    def find_ports():
        """Scan standard + hidden COM ports (e.g. com0com virtual ports)."""
        ports = [p.device for p in
                 serial.tools.list_ports.comports(include_links=True)]
        for i in range(1, 21):
            port = f"COM{i}"
            if port in ports:
                continue
            try:
                s = serial.Serial(port, timeout=0.1)
                s.close()
                ports.append(port)
            except serial.SerialException:
                pass
        return sorted(set(ports))

    def connect(self):
        try:
            self.ser = serial.Serial(
                port=self.port, baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE, timeout=1)
            self.connected = True
            print(f"[gate] Connected to {self.port} @ {self.baudrate} baud")
            return True
        except serial.SerialException as e:
            print(f"[gate] Connection failed: {e}")
            self.connected = False
            return False

    def disconnect(self):
        self._cancel_timer()
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.connected = False
        print("[gate] Disconnected")

    def open_gate(self, plate=None):
        with self._lock:
            if not self.connected or not self.ser.is_open:
                print("[gate] Cannot open — not connected")
                return False
            try:
                self.ser.write(self.open_cmd)
                self.ser.flush()
                print(f"[gate] OPEN sent → {self.open_cmd!r}"
                      + (f"  plate={plate}" if plate else ""))
                self._cancel_timer()
                self._close_timer = threading.Timer(
                    self.open_duration, self._auto_close)
                self._close_timer.daemon = True
                self._close_timer.start()
                return True
            except serial.SerialException as e:
                print(f"[gate] Send error: {e}")
                return False

    def _auto_close(self):
        with self._lock:
            if not self.connected or not self.ser.is_open:
                return
            try:
                self.ser.write(self.close_cmd)
                self.ser.flush()
                print(f"[gate] Auto-CLOSE sent → {self.close_cmd!r}")
            except serial.SerialException as e:
                print(f"[gate] Auto-close error: {e}")

    def _cancel_timer(self):
        if self._close_timer and self._close_timer.is_alive():
            self._close_timer.cancel()