"""Import a Twilio or Telnyx number into Vapi so the agent can place outbound calls.

    uv run --with httpx vapi/import_number.py

.env (one provider):
  TELNYX_API_KEY=KEY...            TELNYX_PHONE_NUMBER=+1...   (use a NEW number: importing re-routes its calls to Vapi)
  or TWILIO_ACCOUNT_SID=AC...      TWILIO_AUTH_TOKEN=...       TWILIO_PHONE_NUMBER=+1...

Vapi-bought numbers have a daily outbound limit, so outbound goes through this number. Inbound
on it routes to the same assistant-request endpoint. For Telnyx, this also adds Vapi's
connection to a Telnyx Outbound Voice Profile (required for outbound). Writes
VAPI_OUTBOUND_PHONE_NUMBER_ID to .env.
"""

import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv, set_key

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"
load_dotenv(ENV)


def env(name: str) -> str | None:
    v = os.getenv(name)
    return v.strip().strip("'\"") if v else None


def die(r: httpx.Response, what: str) -> None:
    if r.is_error:
        sys.exit(f"{what} {r.status_code}: {r.text[:500]}")


vapi = httpx.Client(base_url="https://api.vapi.ai", headers={"Authorization": f"Bearer {env('VAPI_API_KEY')}"}, timeout=30)
server = {"url": env("VOICE_BRIDGE_URL").replace("/vapi/tts", "/vapi/assistant-request"), "timeoutSeconds": 7}

if env("TELNYX_API_KEY"):
    provider, number = "telnyx", env("TELNYX_PHONE_NUMBER")
else:
    provider, number = "twilio", env("TWILIO_PHONE_NUMBER")
if not number:
    sys.exit("Set TELNYX_API_KEY + TELNYX_PHONE_NUMBER (or the TWILIO_* trio) in .env")

existing = next((p for p in vapi.get("/phone-number").json() if p.get("number") == number), None)
if existing:
    r = vapi.patch(f"/phone-number/{existing['id']}", json={"server": server})
    die(r, "Vapi update")
elif provider == "telnyx":
    cred = vapi.post("/credential", json={"provider": "telnyx", "apiKey": env("TELNYX_API_KEY"), "name": "telnyx-demo"})
    die(cred, "Vapi credential")
    r = vapi.post("/phone-number", json={
        "provider": "telnyx", "number": number, "credentialId": cred.json()["id"],
        "name": "Michael outbound line", "server": server,
    })
    die(r, "Vapi import")
else:
    r = vapi.post("/phone-number", json={
        "provider": "twilio", "number": number, "twilioAccountSid": env("TWILIO_ACCOUNT_SID"),
        "twilioAuthToken": env("TWILIO_AUTH_TOKEN"), "name": "Michael outbound line", "server": server,
    })
    die(r, "Vapi import")
phone = r.json()
set_key(str(ENV), "VAPI_OUTBOUND_PHONE_NUMBER_ID", phone["id"], quote_mode="never")
print(f"Vapi: {phone['number']} -> {phone['id']} ({provider}, status {phone.get('status')})")

if provider == "telnyx":
    # Outbound needs the number's connection (now Vapi's) on an Outbound Voice Profile.
    tx = httpx.Client(base_url="https://api.telnyx.com/v2", headers={"Authorization": f"Bearer {env('TELNYX_API_KEY')}"}, timeout=30)
    nums = tx.get("/phone_numbers", params={"filter[phone_number]": number})
    die(nums, "Telnyx number lookup")
    conn_id = (nums.json().get("data") or [{}])[0].get("connection_id")
    if not conn_id:
        sys.exit("Telnyx number has no connection yet; rerun in a minute (Vapi assigns one on import).")
    profiles = tx.get("/outbound_voice_profiles").json().get("data") or []
    profile = next((p for p in profiles if p.get("name") == "vapi-demo"), None)
    if profile is None:
        pr = tx.post("/outbound_voice_profiles", json={"name": "vapi-demo", "whitelisted_destinations": ["US", "CA"]})
        die(pr, "Telnyx create outbound profile")
        profile = pr.json()["data"]
    # Attach the profile to the connection Vapi created (credential or FQDN connection).
    for kind in ("call_control_applications", "credential_connections", "fqdn_connections", "ip_connections"):
        cr = tx.patch(f"/{kind}/{conn_id}", json={"outbound": {"outbound_voice_profile_id": profile["id"]}})
        if cr.status_code != 404:
            die(cr, f"Telnyx attach profile ({kind})")
            print(f"Telnyx: connection {conn_id} ({kind}) -> outbound profile {profile['id']}")
            break
    else:
        sys.exit(f"Couldn't find Telnyx connection {conn_id}; add it to an Outbound Voice Profile in the portal.")

print("Saved VAPI_OUTBOUND_PHONE_NUMBER_ID to .env. Outbound calls now go through this number.")
