#!/usr/bin/env python3
"""
LAST SIGNAL - MobileSAM + Omni 视觉模型 Mask 生成器

严格模式: 无降级策略。所有模型必须可用，否则直接报错退出。

流程:
  1. Omni 视觉模型识别场景中物体 → bbox (可跳过, 使用预定义 bbox)
  2. MobileSAM (ONNX encoder + decoder) 基于 bbox 生成精确 mask
  3. 叠层验证: Omni 审查生成的 mask 与原图是否一致

依赖: onnxruntime, pillow, numpy, opencv-python-headless
模型: models/mobilesam.encoder.onnx + models/mobile_sam.onnx (必须存在)
"""

import cv2
import numpy as np
import json
import os
import sys
import subprocess
import time
from pathlib import Path

from PIL import Image
import onnxruntime as ort

# ── 常量 ──
IMG_W, IMG_H = 940, 627
GAME_W, GAME_H = 960, 640
MOBILESAM_SIZE = 1024  # MobileSAM 标准输入尺寸

# ImageNet 归一化参数
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
# ── 自动生成的 spawn 位置（从 walkable mask 随机采样） ──
_computed_spawns = {}

def _random_walkable_spawn(mask_path):
    """从 walkable mask 中随机选取一个白色像素，返回归一化坐标 [x, y]."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return [0.5, 0.75]
    ys, xs = np.where(mask > 128)
    if len(xs) == 0:
        print(f"    ⚠️  walkable mask 全黑，使用默认 spawn")
        return [0.5, 0.75]
    idx = np.random.randint(len(xs))
    nx = float(xs[idx]) / mask.shape[1]
    ny = float(ys[idx]) / mask.shape[0]
    # 留 2% 边距，避免贴边
    nx = max(0.02, min(0.98, nx))
    ny = max(0.02, min(0.98, ny))
    return [round(nx, 4), round(ny, 4)]

MASK_DIR = os.path.join(ASSETS_DIR, "masks")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MASK_DIR, exist_ok=True)

# ── Omni API ──
MIMO_API_SCRIPT = os.path.expanduser("~/.openclaw/skills/mimo-omni/mimo_api.sh")

# ── MobileSAM 模型路径 (必须存在) ──
ENCODER_PATH = os.path.join(MODELS_DIR, "mobilesam.encoder.onnx")
DECODER_PATH = os.path.join(MODELS_DIR, "mobile_sam.onnx")


# ════════════════════════════════════════════════
# 场景定义
# ════════════════════════════════════════════════


SCENES = {
    "rainy_alley": {
        "image": "rainy_alley_f0.png",
        "objects": [
            {"id": "cardboard_box", "label": "纸箱", "bbox": [100, 450, 250, 550]},
            {"id": "lantern", "label": "灯笼", "bbox": [600, 280, 700, 350]},
            {"id": "bookshop_door", "label": "书店灯光", "bbox": [580, 300, 780, 500]},
        ],
        "walkable": {"bbox": [50, 350, 900, 600], "label": "巷子地面"},
        "water": {"bbox": [50, 500, 900, 600], "label": "路面积水"},
    },
    "bookshop": {
        "image": "bookshop_f0.png",
        "objects": [
            {"id": "bookshelf", "label": "书架", "bbox": [50, 100, 250, 400]},
            {"id": "fireplace", "label": "壁炉", "bbox": [350, 350, 550, 500]},
            {"id": "garden_door", "label": "花园门", "bbox": [700, 150, 900, 500]},
            {"id": "cat_bed", "label": "猫床", "bbox": [400, 400, 550, 500]},
        ],
        "walkable": {"bbox": [30, 200, 930, 600], "label": "书店地面"},
        "water": None,
    },
    "garden": {
        "image": "garden_f0.png",
        "objects": [
            {"id": "butterfly_spot", "label": "蝴蝶", "bbox": [300, 100, 500, 300]},
            {"id": "bird_tree", "label": "小鸟的树", "bbox": [600, 50, 750, 200]},
            {"id": "pond", "label": "小池塘", "bbox": [200, 400, 500, 580]},
            {"id": "bridge_path", "label": "通往小桥", "bbox": [750, 350, 900, 550]},
        ],
        "walkable": {"bbox": [50, 150, 900, 600], "label": "花园地面"},
        "water": {"bbox": [200, 420, 500, 570], "label": "池塘"},
    },
    "bridge": {
        "image": "bridge_f0.png",
        "objects": [
            {"id": "stream", "label": "小溪", "bbox": [200, 380, 700, 550]},
            {"id": "owl_tree", "label": "大树", "bbox": [100, 50, 300, 300]},
            {"id": "rooftop_path", "label": "去高处", "bbox": [700, 100, 900, 400]},
        ],
        "walkable": {"bbox": [50, 200, 900, 400], "label": "桥面"},
        "water": {"bbox": [150, 400, 750, 580], "label": "溪水"},
    },
    "rooftop": {
        "image": "rooftop_f0.png",
        "objects": [
            {"id": "telescope", "label": "望远镜", "bbox": [350, 200, 600, 450]},
            {"id": "blankets", "label": "毯子和靠垫", "bbox": [100, 350, 300, 500]},
        ],
        "walkable": {"bbox": [50, 150, 900, 600], "label": "屋顶地面"},
        "water": None,
    },
}
def omni_analyze_image(image_path, prompt, max_tokens=4096):
    """调用 mimo-omni 分析图片."""
    if not os.path.exists(MIMO_API_SCRIPT):
        raise FileNotFoundError(
            f"mimo-omni 不可用: {MIMO_API_SCRIPT} 不存在\n"
            f"请确保 OpenClaw mimo-omni skill 已安装。")

    result = subprocess.run(
        ["bash", MIMO_API_SCRIPT, "image", image_path, prompt,
         "--max-tokens", str(max_tokens)],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        raise RuntimeError(f"Omni 调用失败 (exit {result.returncode}): {result.stderr}")
    if not result.stdout.strip():
        raise RuntimeError("Omni 返回空结果")
    return result.stdout.strip()


def omni_verify_mask(image_path, mask_path, obj_label, scene_id):
    """
    叠层验证: Omni 审查 mask 与原图是否一致.
    返回: (passed: bool, reason: str)
    """
    img = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"无法读取图片: {image_path}")
    if mask is None:
        raise RuntimeError(f"无法读取 mask: {mask_path}")

    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    # 红色半透明叠加
    overlay = img.copy()
    overlay[mask > 128] = [0, 0, 200]
    blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

    preview_path = os.path.join(MASK_DIR, "_verify_preview.png")
    cv2.imwrite(preview_path, blended)

    prompt = (
        f"这是游戏场景 '{scene_id}' 的物体 '{obj_label}' 的 mask 验证图。\n"
        "红色半透明区域是生成的 mask。\n\n"
        "请判断:\n"
        "1. mask 是否准确覆盖了目标物体? (是/否)\n"
        "2. mask 是否包含过多背景区域? (是/否)\n"
        "3. mask 是否遗漏了物体的重要部分? (是/否)\n\n"
        "如果 mask 质量合格, 回复: PASS\n"
        "如果 mask 质量不合格, 回复: FAIL <原因>\n"
        "只回复 PASS 或 FAIL, 不要其他文字。"
    )
    response = omni_analyze_image(preview_path, prompt)

    if "PASS" in response.upper():
        return True, "验证通过"
    else:
        reason = response.replace("FAIL", "").strip()
        return False, reason


# ════════════════════════════════════════════════
# 场景处理
# ════════════════════════════════════════════════

def find_scene_image(image_name):
    """查找场景图片 (支持 png/webp)."""
    base = os.path.splitext(image_name)[0]
    for ext in [".png", ".webp"]:
        path = os.path.join(ASSETS_DIR, base + ext)
        if os.path.exists(path):
            return path
    return None


def add_edge_transition(mask, zone, size, w, h):
    if zone == "bottom":
        mask[h - size:h, :] = 255
    elif zone == "top":
        mask[0:size, :] = 255
    elif zone == "left":
        mask[:, 0:size] = 255
    elif zone == "right":
        mask[:, w - size:w] = 255


def process_scene(scene_id, data, sam, skip_omni_detect=False):
    """处理单个场景. 必须成功, 失败则抛异常."""
    img_path = find_scene_image(data["image"])
    if not img_path:
        raise FileNotFoundError(f"场景 {scene_id}: 图片不存在 {data['image']}")

    img = cv2.imread(img_path)
    if img is None:
        raise RuntimeError(f"场景 {scene_id}: 无法读取 {img_path}")

    h, w = img.shape[:2]
    combined = np.zeros((h, w), np.uint8)

    # ── Step 1: Omni 识别 (可选) ──
    objects = data["objects"]
    if not skip_omni_detect and objects:
        print(f"  🔍 Omni 识别物体...")
        import re
        prompt = (
            "这张图是 2D 冒险游戏的场景背景 (940x627 像素)。\n"
            "请识别所有可交互的物体/区域，返回 JSON 数组:\n"
            '[{"id": "英文标识", "label": "中文名", "bbox": [x1, y1, x2, y2]}]\n'
            "只返回 JSON, 不要其他文字。bbox 是像素坐标。"
        )
        response = omni_analyze_image(img_path, prompt)
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            detected = json.loads(json_match.group())
            print(f"  ✓ Omni 识别到 {len(detected)} 个物体")
            for obj in objects:
                for det in detected:
                    if det.get("id", "").lower() == obj["id"].lower():
                        obj["bbox"] = det["bbox"]
                        break

    # ── Step 2: MobileSAM 分割每个物体 ──
    for obj in objects:
        bbox = obj["bbox"]
        label = obj["label"]
        obj_id = obj["id"]

        seg = sam.segment(img, bbox)
        ratio = np.count_nonzero(seg) / (h * w) * 100

        # 质量检查 (不是降级, 是精度保障)
        if ratio < 0.05:
            print(f"    ⚠️  {label}: mask 面积过小 ({ratio:.2f}%), bbox 可能不准")
        elif ratio > 35:
            print(f"    ⚠️  {label}: mask 面积过大 ({ratio:.1f}%), 尝试缩小 bbox")
            x1, y1, x2, y2 = bbox
            mx, my = int((x2 - x1) * 0.2), int((y2 - y1) * 0.2)
            seg = sam.segment(img, [x1 + mx, y1 + my, x2 - mx, y2 - my])

        obj_game = cv2.resize(seg, (GAME_W, GAME_H), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(MASK_DIR, f"{scene_id}_{obj_id}_mask.png"), obj_game)
        combined = cv2.bitwise_or(combined, seg)
        print(f"  🎯 {label} ({obj_id}): {ratio:.1f}%")

    # ── Walkable ──
    walkable_cfg = data.get("walkable")
    if walkable_cfg:
        seg = sam.segment(img, walkable_cfg["bbox"])
        ratio = np.count_nonzero(seg) / (h * w) * 100
        if ratio < 1.0:
            print(f"    ⚠️  walkable: mask 太小 ({ratio:.2f}%), bbox 可能不准")
        # 减去障碍物
        for obj in data["objects"]:
            obj_mask_path = os.path.join(MASK_DIR, f"{scene_id}_{obj['id']}_mask.png")
            if os.path.exists(obj_mask_path):
                obs = cv2.imread(obj_mask_path, cv2.IMREAD_GRAYSCALE)
                if obs is not None:
                    obs = cv2.resize(obs, (w, h), interpolation=cv2.INTER_NEAREST)
                    seg[obs > 128] = 0
        seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        walkable_game = cv2.resize(seg, (GAME_W, GAME_H), interpolation=cv2.INTER_NEAREST)
        walkable_path = os.path.join(MASK_DIR, f"{scene_id}_walkable_mask.png")
        cv2.imwrite(walkable_path, walkable_game)
        walk_pct = np.count_nonzero(walkable_game) / (GAME_W * GAME_H) * 100
        print(f"  🚶 walkable ({walkable_cfg['label']}): {walk_pct:.1f}%")

        # 从 walkable mask 随机采样 spawn 位置
        spawn = _random_walkable_spawn(walkable_path)
        _computed_spawns[scene_id] = spawn
        print(f"  🎲 spawn: ({spawn[0]}, {spawn[1]}) (random from walkable)")

    # ── Water ──
    water_cfg = data.get("water")
    if water_cfg:
        seg = sam.segment(img, water_cfg["bbox"])
        ratio = np.count_nonzero(seg) / (h * w) * 100
        if ratio < 0.3:
            print(f"    ⚠️  water: mask 太小 ({ratio:.2f}%), bbox 可能不准")
        seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        seg = cv2.morphologyEx(seg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        water_game = cv2.resize(seg, (GAME_W, GAME_H), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(MASK_DIR, f"{scene_id}_water_mask.png"), water_game)
        water_pct = np.count_nonzero(water_game) / (GAME_W * GAME_H) * 100
        print(f"  💧 water ({water_cfg['label']}): {water_pct:.1f}%")

    # ── 边缘过渡 ──
    for edge in data.get("edge_transitions", []):
        add_edge_transition(combined, edge["zone"], edge["size"], w, h)
        print(f"  🚪 边缘过渡: {edge['label']} ({edge['zone']})")

    # ── 组合 mask ──
    combined_game = cv2.resize(combined, (GAME_W, GAME_H), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(MASK_DIR, f"{scene_id}_mask.png"), combined_game)
    total_pct = np.count_nonzero(combined_game) / (GAME_W * GAME_H) * 100
    print(f"  💾 {scene_id}_mask.png ({total_pct:.1f}% 覆盖)")

    # ── Step 3: 叠层验证 ──
    if objects:
        print(f"  🔎 叠层验证 ({len(objects)} 个物体)...")
        for obj in objects:
            mask_path = os.path.join(MASK_DIR, f"{scene_id}_{obj['id']}_mask.png")
            if os.path.exists(mask_path):
                passed, reason = omni_verify_mask(img_path, mask_path,
                                                   obj["label"], scene_id)
                status = "✅" if passed else "❌"
                print(f"    {status} {obj['label']}: {reason}")


def save_metadata():
    """保存 mask_metadata.json."""
    meta = {}
    for sid, sd in SCENES.items():
        objects = [
            {"id": o["id"], "label": o["label"],
             "mask": f"masks/{sid}_{o['id']}_mask.webp"}
            for o in sd["objects"]
        ]
        if sd.get("walkable"):
            objects.append({
                "id": "walkable", "label": sd["walkable"]["label"],
                "mask": f"masks/{sid}_walkable_mask.webp",
                "type": "walkable"
            })
        if sd.get("water"):
            objects.append({
                "id": "water", "label": sd["water"]["label"],
                "mask": f"masks/{sid}_water_mask.webp",
                "type": "water"
            })
        meta[sid] = {
            "spawn": _computed_spawns.get(sid, [0.5, 0.75]),
            "objects": objects,
            "edge_transitions": [
                {"id": e["id"], "label": e["label"],
                 "zone": e["zone"], "target": e.get("target", "")}
                for e in sd.get("edge_transitions", [])
            ]
        }
    with open(os.path.join(MASK_DIR, "mask_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def git_push_mask(scene_id, obj_id=None):
    """提交并推送."""
    msg = f"mask: {scene_id}/{obj_id}" if obj_id else f"mask: {scene_id}"
    subprocess.run(["git", "add", "-A"], cwd=BASE_DIR, check=True)
    subprocess.run(["git", "commit", "-m", msg], cwd=BASE_DIR, check=True)
    subprocess.run(["git", "push"], cwd=BASE_DIR, check=True)
    print(f"  📤 pushed: {msg}")


# ════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LAST SIGNAL - MobileSAM Mask 生成器")
    parser.add_argument("--scene", help="只处理指定场景")
    parser.add_argument("--skip-omni-detect", action="store_true",
                        help="跳过 Omni 物体识别, 直接用 SCENES 中的 bbox")
    parser.add_argument("--skip-verify", action="store_true",
                        help="跳过 Omni 叠层验证")
    parser.add_argument("--no-push", action="store_true",
                        help="不自动 git push")
    args = parser.parse_args()

    print("=" * 60)
    print("🎭 LAST SIGNAL - MobileSAM + Omni Mask 生成器")
    print("=" * 60)

    # ── 加载 MobileSAM (必须成功) ──
    print("\n🔧 加载模型...")
    sam = MobileSAM()

    # ── 检查 Omni ──
    if not os.path.exists(MIMO_API_SCRIPT):
        raise FileNotFoundError(
            f"mimo-omni 不可用: {MIMO_API_SCRIPT}\n"
            f"请确保 OpenClaw mimo-omni skill 已安装。")
    print(f"  ✓ Omni: {MIMO_API_SCRIPT}")

    # ── 处理场景 ──
    scenes_to_process = SCENES
    if args.scene:
        if args.scene not in SCENES:
            print(f"❌ 未知场景: {args.scene}")
            print(f"   可用: {', '.join(SCENES.keys())}")
            sys.exit(1)
        scenes_to_process = {args.scene: SCENES[args.scene]}

    for scene_id, data in scenes_to_process.items():
        print(f"\n{'─' * 50}")
        print(f"🎬 {scene_id}")
        print(f"{'─' * 50}")

        process_scene(scene_id, data, sam,
                       skip_omni_detect=args.skip_omni_detect)

        if not args.no_push:
            git_push_mask(scene_id)

    # ── 元数据 ──
    save_metadata()
    if not args.no_push:
        git_push_mask("_metadata")

    total_obj = sum(len(s["objects"]) for s in SCENES.values())
    total_edge = sum(len(s.get("edge_transitions", [])) for s in SCENES.values())
    print(f"\n{'=' * 60}")
    print(f"✅ 完成: {total_obj} 个物体 + {total_edge} 个边缘过渡")
    print(f"📁 {MASK_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
