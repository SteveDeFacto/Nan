# ipfs_fetch.py — fetch a wasm by IPFS CID and VERIFY the bytes match the CID.
#
# Why verify: the enclave fetches from an IPFS gateway it does not trust (it may be
# operator-run, e.g. ipfs.nan.host). Content-addressing only helps if we re-check it:
# we pull the whole DAG as a CAR (trustless gateway format), confirm every block
# hashes to its own CID, then reassemble the file from the block referenced by the
# CID the caller asked for. A gateway that swaps or corrupts bytes fails the hash
# check, so "what ran == this exact CID" holds without trusting the gateway.
#
# Supports the CIDs our upload path produces (Kubo `add --cid-version=1`): CIDv1 with
# sha2-256, raw-codec leaves, and dag-pb UnixFS file nodes. sha2-256 only (by design).
#
# Pure stdlib.

import base64
import hashlib
import urllib.request

RAW_CODEC = 0x55
DAGPB_CODEC = 0x70
SHA2_256 = 0x12

_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _uvarint(buf, pos):
    shift = 0
    result = 0
    while True:
        b = buf[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7
        if shift > 63:
            raise ValueError("varint too long")


def _b58decode(s):
    num = 0
    for ch in s:
        num = num * 58 + _B58.index(ch)
    out = num.to_bytes((num.bit_length() + 7) // 8, "big") if num else b""
    pad = len(s) - len(s.lstrip("1"))
    return b"\x00" * pad + out


def cid_str_to_bytes(cid):
    """Decode a multibase CID string to its binary form (the CAR/link key form)."""
    if cid.startswith("Qm"):                     # CIDv0, implicit base58btc
        return _b58decode(cid)
    mb, body = cid[0], cid[1:]
    if mb in ("b", "B"):                         # base32
        pad = "=" * ((8 - len(body) % 8) % 8)
        return base64.b32decode(body.upper() + pad)
    if mb == "z":                                # base58btc
        return _b58decode(body)
    raise ValueError("unsupported CID multibase %r" % mb)


def _cid_info(buf, pos):
    """Parse a binary CID at buf[pos:]; return (cid_bytes, info, newpos)."""
    start = pos
    if buf[pos] == 0x12 and buf[pos + 1] == 0x20:                 # CIDv0 (sha256 multihash)
        cid = bytes(buf[pos:pos + 34])
        return cid, {"codec": DAGPB_CODEC, "mh_code": SHA2_256, "digest": cid[2:34]}, pos + 34
    ver, pos = _uvarint(buf, pos)
    if ver != 1:
        raise ValueError("unsupported CID version %d" % ver)
    codec, pos = _uvarint(buf, pos)
    mh_code, pos = _uvarint(buf, pos)
    mh_len, pos = _uvarint(buf, pos)
    digest = bytes(buf[pos:pos + mh_len]); pos += mh_len
    return bytes(buf[start:pos]), {"codec": codec, "mh_code": mh_code, "digest": digest}, pos


def _fields(msg):
    """Minimal protobuf reader -> list of (field_number, wire_type, value)."""
    out = []
    pos, n = 0, len(msg)
    while pos < n:
        tag, pos = _uvarint(msg, pos)
        field, wt = tag >> 3, tag & 7
        if wt == 0:
            val, pos = _uvarint(msg, pos)
        elif wt == 2:
            ln, pos = _uvarint(msg, pos)
            val = msg[pos:pos + ln]; pos += ln
        elif wt == 1:
            val = msg[pos:pos + 8]; pos += 8
        elif wt == 5:
            val = msg[pos:pos + 4]; pos += 4
        else:
            raise ValueError("bad protobuf wire type %d" % wt)
        out.append((field, wt, val))
    return out


def _parse_dagpb(block):
    """dag-pb PBNode -> (ordered link-CID list, Data bytes). PBNode: 1=Data, 2=Links."""
    links, data = [], b""
    for field, wt, val in _fields(block):
        if field == 1 and wt == 2:
            data = val
        elif field == 2 and wt == 2:            # PBLink: 1=Hash(CID)
            for lf, lwt, lval in _fields(val):
                if lf == 1 and lwt == 2:
                    links.append(lval)
                    break
    return links, data


def _unixfs_data(data):
    """UnixFS Data message -> its Data field (field 2), the file bytes of a dag-pb leaf."""
    for field, wt, val in _fields(data):
        if field == 2 and wt == 2:
            return val
    return b""


def parse_car(car):
    """CARv1 -> {cid_bytes: block_bytes}, verifying every block hashes to its CID."""
    blocks = {}
    hlen, pos = _uvarint(car, 0)
    pos += hlen                                  # skip header; we trust the caller's CID, not the header roots
    n = len(car)
    while pos < n:
        blen, pos = _uvarint(car, pos)
        end = pos + blen
        cid, info, pos = _cid_info(car, pos)
        block = bytes(car[pos:end]); pos = end
        if info["mh_code"] != SHA2_256:
            raise ValueError("unsupported hash in block (sha2-256 only)")
        if hashlib.sha256(block).digest() != info["digest"]:
            raise ValueError("block hash mismatch — gateway served tampered/corrupt data")
        blocks[cid] = block
    return blocks


def reconstruct(cid_bytes, blocks, max_bytes):
    """Reassemble the file rooted at cid_bytes from the verified block map."""
    out = bytearray()

    def walk(c):
        if c not in blocks:
            raise ValueError("CAR is missing block %s" % c.hex())
        _, info, _ = _cid_info(c, 0)
        block = blocks[c]
        if info["codec"] == RAW_CODEC:
            out.extend(block)
        elif info["codec"] == DAGPB_CODEC:
            links, data = _parse_dagpb(block)
            if links:
                for link in links:
                    walk(link)
            else:
                out.extend(_unixfs_data(data))
        else:
            raise ValueError("unsupported CID codec 0x%x" % info["codec"])
        if len(out) > max_bytes:
            raise ValueError("reconstructed file exceeds max %d bytes" % max_bytes)

    walk(cid_bytes)
    return bytes(out)


def verify_car(cid_str, car, max_bytes):
    """Verify a CAR against cid_str and return the file bytes (raises on any mismatch)."""
    cid_bytes = cid_str_to_bytes(cid_str)
    blocks = parse_car(car)
    if cid_bytes not in blocks:
        raise ValueError("CAR does not contain the requested CID %s" % cid_str)
    return reconstruct(cid_bytes, blocks, max_bytes)


def fetch_verified(cid_str, gateway, max_bytes, timeout=120):
    """Fetch the DAG for cid_str from `gateway` as a CAR and return verified file bytes."""
    url = gateway.rstrip("/") + "/ipfs/" + cid_str + "?format=car&dag-scope=all"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.ipld.car"})
    cap = max_bytes + max_bytes // 2 + (1 << 20)     # CAR framing/CID overhead headroom
    with urllib.request.urlopen(req, timeout=timeout) as r:
        car = r.read(cap + 1)
    if len(car) > cap:
        raise ValueError("CAR larger than allowed (%d bytes cap)" % cap)
    return verify_car(cid_str, car, max_bytes)
