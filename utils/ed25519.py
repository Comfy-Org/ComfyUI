"""Pure-Python Ed25519 verification using the RFC 8032 group equation."""

import hashlib


_P = 2**255 - 19
_L = 2**252 + 27742317777372353535851937790883648493
_D = -121665 * pow(121666, _P - 2, _P) % _P
_SQRT_M1 = pow(2, (_P - 1) // 4, _P)
_IDENTITY = (0, 1, 1, 0)


class InvalidSignature(Exception):
    pass


def _add(p, q):
    # Extended Edwards coordinates: x = X/Z, y = Y/Z, xy = T/Z.
    a = (p[1] - p[0]) * (q[1] - q[0]) % _P
    b = (p[1] + p[0]) * (q[1] + q[0]) % _P
    c = 2 * _D * p[3] * q[3] % _P
    d = 2 * p[2] * q[2] % _P
    e, f, g, h = b - a, d - c, d + c, b + a
    return e * f % _P, g * h % _P, f * g % _P, e * h % _P


def _multiply(point, scalar):
    result = _IDENTITY
    while scalar:
        if scalar & 1:
            result = _add(result, point)
        point = _add(point, point)
        scalar >>= 1
    return result


def _decode(data):
    encoded = int.from_bytes(data, "little")
    y, sign = encoded & ((1 << 255) - 1), encoded >> 255
    if y >= _P:
        raise InvalidSignature
    x_squared = (y * y - 1) * pow(_D * y * y + 1, _P - 2, _P) % _P
    x = pow(x_squared, (_P + 3) // 8, _P)
    if x * x % _P != x_squared:
        x = x * _SQRT_M1 % _P
    if x * x % _P != x_squared or (x == 0 and sign):
        raise InvalidSignature
    if x & 1 != sign:
        x = _P - x
    return x, y, 1, x * y % _P


_BASE = _decode(bytes.fromhex("5866666666666666666666666666666666666666666666666666666666666666"))


class Ed25519PublicKey:
    def __init__(self, data):
        self._data = memoryview(data).tobytes()
        if len(self._data) != 32:
            raise ValueError("An Ed25519 public key is 32 bytes long")

    @classmethod
    def from_public_bytes(cls, data):
        return cls(data)

    def verify(self, signature, data):
        signature = memoryview(signature).tobytes()
        if len(signature) != 64:
            raise InvalidSignature
        scalar = int.from_bytes(signature[32:], "little")
        if scalar >= _L:
            raise InvalidSignature
        public_point = _decode(self._data)
        r = _decode(signature[:32])
        digest = hashlib.sha512(signature[:32] + self._data)
        digest.update(data)
        challenge = int.from_bytes(digest.digest(), "little") % _L
        left = _multiply(_BASE, scalar)
        right = _add(r, _multiply(public_point, challenge))
        if (left[0] * right[2] - right[0] * left[2]) % _P or (left[1] * right[2] - right[1] * left[2]) % _P:
            raise InvalidSignature
