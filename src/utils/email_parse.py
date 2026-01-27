import html
import re
from email import policy
from email.parser import BytesParser

from src.utils.io import decode_bytes

_TAG_RE = re.compile(r"<[^>]+>")


def html_to_text(text: str) -> str:
    if not text:
        return ""
    text = _TAG_RE.sub(" ", text)
    return html.unescape(text)


def _extract_part_text(part) -> str:
    if part.get_content_maintype() == "multipart":
        return ""
    if part.get_content_disposition() == "attachment":
        return ""
    payload = part.get_payload(decode=True)
    if payload is None:
        text = part.get_payload()
        return text if isinstance(text, str) else ""
    return decode_bytes(payload)


def parse_email_bytes(raw_bytes: bytes) -> tuple[str, str]:
    msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)
    subject = msg.get("subject") or ""
    body_text = ""
    if msg.is_multipart():
        plain_parts = []
        html_parts = []
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                plain_parts.append(_extract_part_text(part))
            elif content_type == "text/html":
                html_parts.append(_extract_part_text(part))
        if plain_parts:
            body_text = "\n".join(p for p in plain_parts if p)
        elif html_parts:
            body_text = html_to_text("\n".join(p for p in html_parts if p))
    else:
        body_text = _extract_part_text(msg)
    return subject, body_text
