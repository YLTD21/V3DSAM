#!/usr/bin/env python3
"""Generate the V3DSAM framework figure (fig1) via SiliconFlow image API.

Standalone (stdlib only). Does NOT touch v3dsam source; does NOT use the
OpenRouter-specific nature-skills script (that one targets openrouter.ai, but
the user's key is SiliconFlow). Same idea, correct provider.
"""
import argparse, json, os, sys, urllib.request

API = "https://api.siliconflow.cn/v1/images/generations"
KEY_ENV = "SILICONFLOW_API_KEY"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen-Image")
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--size", default="1152x448")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    key = os.environ.get(KEY_ENV)
    if not key:
        sys.exit(f"set {KEY_ENV} (SiliconFlow key) first")

    prompt = open(args.prompt_file, encoding="utf-8").read().strip()
    payload = {
        "model": args.model,
        "prompt": prompt,
        "image_size": args.size,
        "batch_size": args.batch,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance,
        "prompt_enhancement": True,
    }
    req = urllib.request.Request(
        API,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    print(f"[gen] POST {API} model={args.model} size={args.size} ...")
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            resp = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        sys.exit(f"[gen] HTTP {e.code}: {e.read().decode('utf-8', 'replace')}")

    # SiliconFlow returns both `images` and `data`; grab the first url from either.
    url = None
    for arr in (resp.get("images"), resp.get("data")):
        if arr and arr[0].get("url"):
            url = arr[0]["url"]
            break
    if not url:
        sys.exit(f"[gen] no image url in response: {json.dumps(resp)[:500]}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as r:
        data = r.read()
    with open(args.out, "wb") as f:
        f.write(data)
    print(f"[gen] saved {len(data)/1024:.0f} KB -> {args.out}")
    print(f"[gen] seed={resp.get('seed')} timings={resp.get('timings')}")


if __name__ == "__main__":
    main()
