"""
Local WhatsApp simulation — runs the exact same code path as a real WhatsApp message.
Patches only the Meta network calls (download image, send reply).
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

IMAGE_PATH = r"C:\Users\kenne\Downloads\samplepic2.png"
FAKE_SENDER = "60123456789"


async def main():
    # Patch BEFORE importing orchestrator so the imports inside it pick up the mocks
    import tools.whatsapp_client as wa

    async def mock_download_media(media_id: str) -> bytes:
        print(f"[mock] Downloading image from local file: {IMAGE_PATH}")
        return Path(IMAGE_PATH).read_bytes()

    async def mock_send_text(to: str, text: str) -> None:
        print(f"\n{'='*50}")
        print(f"WhatsApp TEXT reply → {to}")
        print(f"{'='*50}")
        print(text)

    async def mock_send_interactive(to: str, message: dict) -> None:
        print(f"\n{'='*50}")
        print(f"WhatsApp INTERACTIVE reply → {to}")
        print(f"{'='*50}")
        print(json.dumps(message, indent=2))

    wa.download_media = mock_download_media
    wa.send_text = mock_send_text
    wa.send_interactive = mock_send_interactive

    from orchestrator import handle_message

    envelope = {
        "message_id": "wamid.local_test_001",
        "from": FAKE_SENDER,
        "type": "image",
        "text": "",
        "image": {"id": "local_test"},
        "interactive": {},
    }

    print(f"Simulating WhatsApp image message from {FAKE_SENDER}...")
    print(f"Image: {IMAGE_PATH}\n")

    await handle_message(envelope)
    print("\nSimulation complete.")


asyncio.run(main())
