#!/usr/bin/env python3
import re

def decode_bytes(b):
    if not b:
        return ''
    try:
        return b.decode('utf-8')
    except Exception:
        return b.decode('latin-1', errors='ignore')

def encode_str(s):
    if s is None:
        return b''
    return s.encode('utf-8')

def normalize_and_strip_claude(s):
    """Remove any occurrence of 'claude' (case-insensitive) and trim whitespace."""
    if not s:
        return s
    s = re.sub(r'(?i)claude', '', s)
    s = re.sub(r'\s{2,}', ' ', s)
    return s.strip()

def remove_coauthored_by_trailers(msg_b):
    if not msg_b:
        return msg_b
    s = decode_bytes(msg_b)
    s = re.sub(r'(?im)^[ \t]*Co-Authored-By:.*claude.*\n?', '', s)
    s = re.sub(r'\n{2,}', '\n\n', s)
    return encode_str(s)

def commit_callback(commit):
    if getattr(commit, 'author_name', None) is not None:
        name = decode_bytes(commit.author_name)
        name = normalize_and_strip_claude(name)
        commit.author_name = encode_str(name)

    if getattr(commit, 'committer_name', None) is not None:
        name = decode_bytes(commit.committer_name)
        name = normalize_and_strip_claude(name)
        commit.committer_name = encode_str(name)

    if getattr(commit, 'author_email', None) is not None:
        email = decode_bytes(commit.author_email)
        email = normalize_and_strip_claude(email)
        commit.author_email = encode_str(email)

    if getattr(commit, 'committer_email', None) is not None:
        email = decode_bytes(commit.committer_email)
        email = normalize_and_strip_claude(email)
        commit.committer_email = encode_str(email)

    if getattr(commit, 'message', None) is not None:
        commit.message = remove_coauthored_by_trailers(commit.message)
