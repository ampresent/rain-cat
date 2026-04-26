---
name: rain-cat-workflow
description: Rain Cat game development workflow — AI-powered point-and-click adventure game creation with Pollinations.AI and canvas effects. Use when working on the Rain Cat project.
---

# Rain Cat Workflow

Complete development workflow for the 🐱🌧️ Rain Cat game.

## Quick Reference

| Task | Command |
|------|---------|
| Generate base assets | `python3 gen_assets.py` |
| Generate depth + lighting | `python3 gen_depth_lighting.py` |
| Generate masks | `python3 gen_masks.py` |
| Build (PNG→WebP) | `python3 build.py` |
| Local test | `python3 -m http.server 8765` |

## Story

A small orange cat (喵喵) is caught in the rain and must find shelter. Along the way, it meets friends who teach it about the world:

1. **rainy_alley** — Cold, wet, looking for shelter
2. **bookshop** — Meet 灰灰 (shopkeeper cat), warm up by the fire
3. **garden** — Meet 小蓝 (butterfly) and 知更 (bird)
4. **bridge** — Meet 金金 (koi fish) and 月月 (owl)
5. **rooftop** — See the whole world from above, ending

## Characters

- 🐱 **喵喵** (miao) — Small orange tabby, curious, a bit scared
- 📚 **灰灰** (shopkeeper) — Wise old grey cat, bookshop owner
- 🦋 **小蓝** (butterfly) — Blue morpho butterfly, playful
- 🐦 **知更** (bird) — Cheerful robin, helpful
- 🐟 **金金** (fish) — Golden koi, knowledgeable
- 🦉 **月月** (owl) — Wise owl, stargazer

## Environment Setup

```bash
pip3 install --break-system-packages requests pillow
python3 gen_assets.py
python3 -m http.server 8765
```
