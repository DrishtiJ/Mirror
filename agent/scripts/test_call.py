"""Join a LiveKit room as a test caller, optionally speak a WAV, record the agent.

    uv run scripts/test_call.py [--say path.wav] [--seconds 15]

Needs the agent running (`uv run agent.py dev`). Saves out/agent_reply.wav and prints
the agent's transcript and the caller transcript the agent heard.
"""
import argparse, asyncio, json, os, uuid, wave
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from livekit import api, rtc

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


async def main(say: str | None, seconds: float, metadata: str) -> None:
    room_name = f"test-{uuid.uuid4().hex[:6]}"
    lkapi = api.LiveKitAPI()
    await lkapi.room.create_room(api.CreateRoomRequest(name=room_name, metadata=metadata))
    token = (
        api.AccessToken().with_identity("test-caller").with_name("Test Caller")
        .with_grants(api.VideoGrants(room_join=True, room=room_name)).to_jwt()
    )
    room = rtc.Room()
    chunks: list[np.ndarray] = []
    lines: list[str] = []

    @room.on("track_subscribed")
    def on_track(track, pub, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            async def pull():
                async for ev in rtc.AudioStream(track, sample_rate=24000, num_channels=1):
                    chunks.append(np.frombuffer(ev.frame.data, dtype=np.int16).copy())
            asyncio.ensure_future(pull())

    def on_text(reader, participant_identity):
        async def read():
            text = await reader.read_all()
            if reader.info.attributes.get("lk.transcription_final") == "true" and text.strip():
                who = "agent" if participant_identity != "test-caller" else "caller(heard)"
                lines.append(f"{who}: {text.strip()}")
        asyncio.ensure_future(read())

    room.register_text_stream_handler("lk.transcription", on_text)
    await room.connect(os.environ["LIVEKIT_URL"], token)
    print(f"joined {room_name}; waiting for agent...")

    if say:
        await asyncio.sleep(8)  # let the greeting finish
        with wave.open(say) as w:
            sr, pcm = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        src = rtc.AudioSource(sr, 1)
        await room.local_participant.publish_track(
            rtc.LocalAudioTrack.create_audio_track("mic", src), rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )
        step = sr // 100
        for i in range(0, len(pcm) + sr, step):  # audio then 1s of silence
            chunk = pcm[i:i + step]
            chunk = np.pad(chunk, (0, step - len(chunk)))
            await src.capture_frame(rtc.AudioFrame(chunk.tobytes(), sr, 1, step))
        print(f"said {say}")

    await asyncio.sleep(seconds)
    await room.disconnect()
    await lkapi.room.delete_room(api.DeleteRoomRequest(room=room_name))
    await lkapi.aclose()

    out = ROOT / "out" / "agent_reply.wav"
    out.parent.mkdir(exist_ok=True)
    audio = np.concatenate(chunks) if chunks else np.zeros(0, np.int16)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(audio.tobytes())
    voiced = float((np.abs(audio) > 500).mean()) if len(audio) else 0.0
    print(f"recorded {len(audio)/24000:.1f}s of agent audio ({voiced:.0%} voiced) -> {out}")
    print("\n".join(lines) or "(no transcripts)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--say")
    p.add_argument("--seconds", type=float, default=12)
    p.add_argument("--metadata", default=json.dumps({"caller_name": "Test Caller"}))
    a = p.parse_args()
    asyncio.run(main(a.say, a.seconds, a.metadata))
