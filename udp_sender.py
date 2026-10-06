"""Sends robot control packets to the avatar runtimes (JSON over UDP, localhost).

Knows nothing about MediaPipe: it only sees finished control values.  UDP is
deliberate: a lost or late packet is simply replaced by the next one.
"""
import json
import socket
import time

PROTOCOL_VERSION = 1


def make_packet(seq, tracking, calibrating, face, head, note=None):
    """face: {CONTROL: value}; head: (yaw, pitch, roll) in radians, avatar's own left/right.

    ``note`` is free text for the runtime's debug overlay (e.g. the scripted beat being played).
    """
    packet = {
        "v": PROTOCOL_VERSION,
        "seq": seq,
        "timestamp": round(time.time(), 4),
        "tracking": bool(tracking),
        "calibrating": bool(calibrating),
        "head": {"yaw": round(head[0], 5), "pitch": round(head[1], 5), "roll": round(head[2], 5)},
        "face": {name.lower(): round(value, 4) for name, value in face.items()},
    }
    if note:
        packet["note"] = str(note)
    return packet


class UdpSender:
    def __init__(self, targets):
        """targets: [{"host": ..., "port": ..., "name": ...}]"""
        self.targets = [(t["host"], int(t["port"])) for t in targets]
        self.names = [t.get("name", f"{t['host']}:{t['port']}") for t in targets]
        self.sent = 0
        self.errors = 0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, packet):
        data = json.dumps(packet, separators=(",", ":")).encode("utf-8")
        for target in self.targets:
            try:
                self._sock.sendto(data, target)
            except OSError:
                self.errors += 1     # nobody listening is fine; keep going
        self.sent += 1

    def close(self):
        self._sock.close()
