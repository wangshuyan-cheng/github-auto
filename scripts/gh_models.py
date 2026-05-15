#!/usr/bin/env python3
"""
GitHub Models API — 免费 GPT-4o / Llama 405B 等模型调用

用法:
  python3 gh_models.py chat <model> "<prompt>"
  python3 gh_models.py list                    # 列出可用模型
  python3 gh_models.py test                    # 测试连通性

环境变量:
  GH_TOKEN  — GitHub token（有 models:read 权限即可）
             不设置则尝试读取 ~/.config/gh/hosts.yml 中的 token
"""
import os, json, sys, requests
from pathlib import Path

BASE = "https://models.inference.ai.azure.com"

def get_token():
    """获取 GitHub token"""
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        # 从 gh CLI 配置中读取
        gh_config = Path.home() / ".config" / "gh" / "hosts.yml"
        if gh_config.exists():
            for line in gh_config.read_text().splitlines():
                if "oauth_token:" in line:
                    token = line.split(":", 1)[1].strip()
                    break
    return token

def chat(model: str, messages: list, **kwargs):
    """调用聊天模型"""
    token = get_token()
    if not token:
        raise RuntimeError("未找到 GitHub token，请设置 GH_TOKEN 环境变量")
    r = requests.post(
        f"{BASE}/chat/completions",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"model": model, "messages": messages, **kwargs}
    )
    return r.json()

def embed(model: str, input_texts: list):
    """调用嵌入模型"""
    token = get_token()
    r = requests.post(
        f"{BASE}/embeddings",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json={"model": model, "input": input_texts}
    )
    return r.json()

def list_models():
    """列出可用模型"""
    token = get_token()
    r = requests.get(f"{BASE}/models", headers={"Authorization": f"Bearer {token}"})
    models = r.json()
    print("📋 GitHub Models 可用模型：")
    for m in models:
        name = m.get("id", "?")
        print(f"  • {name.split('/')[-1]:40s} → {name}")
    print(f"\n共 {len(models)} 个模型")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "list":
        list_models()

    elif cmd == "chat":
        model = sys.argv[2] if len(sys.argv) > 2 else "gpt-4o-mini"
        prompt = sys.argv[3] if len(sys.argv) > 3 else "say hello"
        result = chat(model, [{"role": "user", "content": prompt}])
        if "choices" in result:
            print(result["choices"][0]["message"]["content"])
        else:
            print(json.dumps(result, indent=2))

    elif cmd == "test":
        try:
            token = get_token()
            if not token:
                print("❌ 未找到 GitHub token")
                sys.exit(1)
            print(f"✅ Token 找到（{token[:8]}...{token[-4:]}）")

            # 测试聊天
            result = chat("gpt-4o-mini", [{"role": "user", "content": "echo hello"}])
            if "choices" in result:
                print(f"✅ 聊天模型 OK: {result['choices'][0]['message']['content']}")
            else:
                print(f"❌ 聊天失败: {json.dumps(result, indent=2)[:200]}")

            # 测试嵌入
            er = embed("text-embedding-3-small", ["hello"])
            if "data" in er:
                print(f"✅ 嵌入模型 OK: 维度={len(er['data'][0]['embedding'])}")
            else:
                print(f"❌ 嵌入失败: {str(er)[:200]}")

            print("\n🎉 GitHub Models 全线正常！")

        except Exception as e:
            print(f"❌ 测试失败: {e}")
            sys.exit(1)

    else:
        print(f"未知命令: {cmd}")
        print(__doc__)
