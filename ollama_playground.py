#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Ollama Playground (local, general-purpose)
#
# Features:
# - Interactive chat loop with your Ollama model (default: llama3.2)
# - Optional system prompt (--system)
# - Attach one or more files as context (--file path ...); content is pasted into the first turn
# - Temperature control (--temp), context tokens (--ctx)
# - JSON mode (--json) to encourage structured output
# - Save transcript to a markdown file (--save chat.md)
# - Commands: /sys, /reset, /save <file>, /model <name>, /exit
#
# Note: This calls "ollama run <model>" each turn and resends local history.
import argparse
import os
import subprocess
import textwrap
from datetime import datetime

def read_files(paths):
    chunks = []
    for p in paths or []:
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            chunks.append(f"\n=== FILE: {os.path.basename(p)} ===\n{content}\n")
        except Exception as e:
            chunks.append(f"\n=== FILE: {os.path.basename(p)} (READ ERROR) ===\n{e}\n")
    return "\n".join(chunks)

def build_prompt(system_msg, history, json_mode=False):
    sys_block = f"SYSTEM:\n{system_msg}\n\n" if system_msg else ""
    conv = []
    for role, content in history:
        conv.append(f"{role.upper()}:\n{content}\n")
    prompt = sys_block + "\n".join(conv)
    if json_mode:
        prompt += "\n(Respond ONLY with valid JSON. No extra text.)\n"
    return prompt

def run_ollama(model, prompt, ctx_tokens=None, temperature=0.7):
    cmd = ["ollama", "run", model]
    env = os.environ.copy()
    if ctx_tokens:
        env["OLLAMA_NUM_CTX"] = str(ctx_tokens)
    env["OLLAMA_TEMPERATURE"] = str(temperature)
    proc = subprocess.run(cmd, input=prompt.encode("utf-8"), capture_output=True, env=env)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))
    return proc.stdout.decode("utf-8", errors="ignore").strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama3.2", help="Ollama model name")
    ap.add_argument("--system", default="", help="System prompt text")
    ap.add_argument("--file", nargs="*", help="Context files (text/markdown/code). Included in first turn.")
    ap.add_argument("--temp", type=float, default=0.7, help="Sampling temperature")
    ap.add_argument("--ctx", type=int, default=None, help="Context window tokens (OLLAMA_NUM_CTX)")
    ap.add_argument("--json", action="store_true", help="Ask model to respond only with JSON")
    ap.add_argument("--save", default="", help="Save transcript to this markdown file")
    args = ap.parse_args()

    history = []  # list of (role, content)
    system_msg = args.system.strip()

    # If files attached, send them as the first "context" from USER
    if args.file:
        files_block = read_files(args.file)
        if files_block:
            history.append(("user", f"CONTEXT FILES BELOW. The assistant should use them when relevant.\n{files_block}"))

    print(f"[Ollama Playground] Model={args.model} | temp={args.temp} | ctx={args.ctx or 'default'} | JSON={args.json}")
    if system_msg:
        print(f"[system] {system_msg}\n")
    print("Type your message. Commands: /sys, /reset, /save, /model, /exit")

    transcript = []
    if system_msg:
        transcript.append(f"### SYSTEM\n{system_msg}\n")

    model = args.model

    while True:
        try:
            user = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user:
            continue

        # Commands
        if user.startswith("/"):
            parts = user.split(" ", 1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""
            if cmd == "/exit":
                print("Bye.")
                break
            elif cmd == "/reset":
                history = []
                print("[reset] conversation cleared.")
                continue
            elif cmd == "/sys":
                system_msg = arg
                print("[system updated]")
                continue
            elif cmd == "/save":
                path = arg or (args.save or f"ollama_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
                transcript_text = "\n".join(transcript)
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(transcript_text)
                    print(f"[saved] {path}")
                except Exception as e:
                    print(f"[save error] {e}")
                continue
            elif cmd == "/model":
                model = arg or model
                print(f"[model set] {model}")
                continue
            else:
                print("Unknown command. Use /sys, /reset, /save <file>, /model <name>, /exit")
                continue

        # Normal user turn
        history.append(("user", user))
        prompt = build_prompt(system_msg, history, json_mode=args.json)
        try:
            reply = run_ollama(model, prompt, ctx_tokens=args.ctx, temperature=args.temp)
        except Exception as e:
            print(f"[ollama error]\n{e}")
            history.pop()  # revert last user message on error
            continue

        history.append(("assistant", reply))
        print(f"\nAssistant> {reply}\n")

        # Append to transcript
        transcript.append(f"**You:** {user}")
        transcript.append(f"**Assistant:**\n{reply}\n")

        # Autosave if --save provided
        if args.save:
            try:
                with open(args.save, "w", encoding="utf-8") as f:
                    f.write("\n".join(transcript))
            except Exception as e:
                print(f"[autosave error] {e}")

if __name__ == "__main__":
    main()
