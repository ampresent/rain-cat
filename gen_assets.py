#!/usr/bin/env python3
"""
🐱🌧️ Rain Cat — 场景与角色素材生成
使用 Pollinations.AI 生成游戏场景背景和角色肖像
"""
import os
import sys
import time
import urllib.parse
import requests

BASE = "https://image.pollinations.ai/prompt/"
OUTPUT_DIR = "assets"
NUM_FRAMES = 5
STYLE = "digital painting, cozy warm colors, Studio Ghibli inspired, soft lighting, children's book illustration, detailed background, 960x640"

FRAME_MODS = [
    "gentle rain, warm lights glowing, ",
    "slightly heavier rain, puddles reflecting lights, ",
    "steady rain, misty atmosphere, ",
    "light rain, clouds parting slightly, ",
    "drizzle stopping, golden hour light peeking through, ",
]

SCENE_PROMPTS = {
    "rainy_alley": (
        "A narrow cobblestone alley in a cozy town during rain, "
        "warm lantern lights reflecting on wet stones, "
        "potted plants under eaves, a small cardboard box with a cat peeking out, "
        "old brick walls with ivy, puddles reflecting amber light, "
        "{style}"
    ),
    "bookshop": (
        "Interior of a cozy old bookshop, warm golden lighting, "
        "floor-to-ceiling bookshelves, a reading nook by the window with rain outside, "
        "a small cat bed near a fireplace, steaming cup of tea on a wooden desk, "
        "fairy lights and dried flowers hanging from ceiling, cat-sized door flap, "
        "{style}"
    ),
    "garden": (
        "A secret garden behind the bookshop after rain stops, "
        "wet flowers glistening, butterflies emerging, "
        "a small pond with lily pads, rainbow in the sky, "
        "wooden bench, climbing roses on a trellis, birdhouse on a post, "
        "{style}"
    ),
    "bridge": (
        "A small stone bridge over a gentle stream at sunset, "
        "koi fish visible in clear water, wildflowers on the banks, "
        "a few fireflies starting to glow, weeping willow tree, "
        "distant town silhouette with warm lights, "
        "{style}"
    ),
    "rooftop": (
        "A cozy rooftop garden at night under stars, "
        "the town spread out below with warm lights, "
        "a telescope on a wooden platform, blankets and cushions, "
        "moonlight casting soft shadows, a sleeping cat curled up, "
        "shooting star in the sky, "
        "{style}"
    ),
}

PORTRAITS = {
    "miao": (
        "Portrait of a small orange tabby cat with big curious eyes, "
        "slightly wet fur, cute expression, Studio Ghibli style, "
        "warm lighting, digital painting, detailed whiskers"
    ),
    "shopkeeper": (
        "Portrait of a wise old grey cat wearing tiny round glasses, "
        "gentle smile, sitting among books, Studio Ghibli style, "
        "warm golden lighting, digital painting"
    ),
    "butterfly": (
        "Portrait of a beautiful blue morpho butterfly with sparkling wings, "
        "friendly expression, Studio Ghibli style, "
        "soft pastel colors, digital painting"
    ),
    "bird": (
        "Portrait of a small cheerful robin with a red chest, "
        "head tilted, curious expression, Studio Ghibli style, "
        "warm lighting, digital painting"
    ),
    "fish": (
        "Portrait of a golden koi fish jumping out of water, "
        "playful expression, water droplets sparkling, Studio Ghibli style, "
        "sunset lighting, digital painting"
    ),
    "moon": (
        "Portrait of a wise old owl with crescent moon markings on chest, "
        "calm serene expression, night sky background, Studio Ghibli style, "
        "moonlight, digital painting"
    ),
}


def download(url, filepath):
    """下载图片"""
    try:
        print(f"  📥 {os.path.basename(filepath)}...", end=" ", flush=True)
        resp = requests.get(url, timeout=120, allow_redirects=True)
        if resp.status_code == 200 and len(resp.content) > 1000:
            with open(filepath, "wb") as f:
                f.write(resp.content)
            print(f"✅ ({len(resp.content)//1024}KB)")
            return True
        else:
            print(f"❌ status={resp.status_code} size={len(resp.content)}")
            return False
    except Exception as e:
        print(f"❌ {e}")
        return False


def gen_scene_base(scene_name, prompt_template):
    """生成场景基础图（VFX 引擎模式）"""
    w, h = 960, 640
    base_seed = 42
    prompt = prompt_template.format(style=STYLE)
    encoded = urllib.parse.quote(prompt)
    url = f"{BASE}{encoded}?width={w}&height={h}&seed={base_seed}&model=flux&nologo=true"

    filename = f"{scene_name}_f0.png"
    filepath = os.path.join(OUTPUT_DIR, filename)
    if download(url, filepath):
        return 1
    time.sleep(2)
    return 0


def gen_portrait(name, prompt):
    """生成角色肖像"""
    w, h = 512, 512
    encoded = urllib.parse.quote(prompt)
    url = f"{BASE}{encoded}?width={w}&height={h}&seed=42&model=flux&nologo=true"
    filepath = os.path.join(OUTPUT_DIR, f"{name}.png")
    return download(url, filepath)


if __name__ == "__main__":
    print("=" * 50)
    print("🐱🌧️ Rain Cat — 场景生成")
    print("   VFX 引擎模式: 每场景 1 张基础图")
    print("=" * 50)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    ok, fail = 0, 0

    # 场景背景
    for scene_name, prompt in SCENE_PROMPTS.items():
        print(f"\n📁 {scene_name}:")
        if gen_scene_base(scene_name, prompt):
            ok += 1
        else:
            fail += 1
        time.sleep(2)

    # 角色肖像
    print(f"\n👤 角色肖像:")
    for name, prompt in PORTRAITS.items():
        if gen_portrait(name, prompt):
            ok += 1
        else:
            fail += 1
        time.sleep(2)

    print(f"\n{'='*50}")
    print(f"✅ 成功: {ok}  ❌ 失败: {fail}")
    print(f"📁 {OUTPUT_DIR}")
