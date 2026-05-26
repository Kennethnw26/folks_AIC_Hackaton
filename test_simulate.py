import asyncio
import base64
import httpx
from pathlib import Path

IMAGE_PATH = r"C:\Users\kenne\Downloads\samplepic2.png"

async def test():
    image_b64 = base64.b64encode(Path(IMAGE_PATH).read_bytes()).decode()
    print(f"Image loaded, base64 length: {len(image_b64)}")

    async with httpx.AsyncClient(timeout=120.0) as c:
        r = await c.post(
            "http://localhost:8000/dev/simulate_image",
            json={"image_b64": image_b64},
        )
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        import json
        data = r.json()
        print("\n=== PROOF ===")
        print(json.dumps(data.get("proof"), indent=2))
        print("\n=== MATCH ===")
        print(json.dumps(data.get("match"), indent=2))
        print("\n=== FRAUD ===")
        print(json.dumps(data.get("fraud"), indent=2))
    else:
        print(f"Error: {r.text[:500]}")

asyncio.run(test())
