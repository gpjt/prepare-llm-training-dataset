import hashlib
import struct


def hash_doc(doc):
    return hashlib.sha256(
        struct.pack(f"<{len(doc)}H", *doc)
    ).hexdigest()
