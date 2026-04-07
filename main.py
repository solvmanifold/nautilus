import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.text import Text

DEFAULT_MODEL_OPENAI = "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
DEFAULT_BASE_URL = "http://localhost:8123/v1"
DEFAULT_MODEL_MLX = "mlx-community/NVIDIA-Nemotron-3-Nano-30B-A3B-4bit"

CHATS_DIR = Path("chats")

GRAY = "\033[90m"
RESET = "\033[0m"

console = Console()


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def extract_answer(text: str) -> str:
    """Return content after </think>, or the full text if no think block present."""
    if "</think>" in text:
        return text[text.index("</think>") + len("</think>"):].strip()
    return text.strip()


def slugify(text: str) -> str:
    # Use only the first line in case the model outputs extra commentary
    text = text.splitlines()[0] if text.splitlines() else text
    text = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    return re.sub(r"[\s_-]+", "_", text)[:60]


def generate_title_openai(client: OpenAI, model: str, history: list) -> str:
    """Ask the model for a concise snake_case title for this chat."""
    lines = []
    for msg in history[:6]:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = strip_thinking(msg["content"])[:300]
        lines.append(f"{role}: {content}")
    snippet = "\n".join(lines)

    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": (
                f"Conversation:\n{snippet}\n\n"
                "Give a snake_case filename for this conversation (3-5 words). "
                "Good examples: cuda_intro, python_async, algebraic_topology. "
                "Output only the filename."
            ),
        }],
        max_tokens=50,
    )
    raw = response.choices[0].message.content or ""
    return slugify(extract_answer(raw)) or "chat"


def generate_title_mlx(model, tokenizer, history: list) -> str:
    from mlx_lm import generate
    lines = []
    for msg in history[:6]:
        role = "User" if msg["role"] == "user" else "Assistant"
        content = strip_thinking(msg["content"])[:300]
        lines.append(f"{role}: {content}")
    snippet = "\n".join(lines)

    prompt_messages = [{
        "role": "user",
        "content": (
            f"Conversation:\n{snippet}\n\n"
            "Give a snake_case filename for this conversation (3-5 words). "
            "Good examples: cuda_intro, python_async, algebraic_topology. "
            "Output only the filename."
        ),
    }]
    formatted = tokenizer.apply_chat_template(
        prompt_messages, tokenize=False, add_generation_prompt=True
    )
    raw = generate(model, tokenizer, prompt=formatted, max_tokens=500, verbose=False)
    return slugify(extract_answer(raw)) or "chat"


def stream_response_openai(stream, show_thinking: bool = True) -> str:
    """
    Phase 1: stream reasoning_content live in gray.
    Phase 2: stream content live as plain text, then snap to rendered markdown.
    """
    reasoning_parts = []
    answer_parts = []

    for chunk in stream:
        delta = chunk.choices[0].delta
        reasoning = getattr(delta, "reasoning_content", None)
        content = delta.content

        if reasoning:
            reasoning_parts.append(reasoning)
            if show_thinking:
                sys.stdout.write(GRAY + reasoning)
                sys.stdout.flush()
        elif content:
            if reasoning_parts and not answer_parts:
                if show_thinking:
                    sys.stdout.write(RESET + "\n")
                    sys.stdout.flush()
            answer_parts.append(content)

    answer = "".join(answer_parts).strip()
    if answer:
        console.print(Markdown(answer))

    return answer


def stream_response_mlx(stream, show_thinking: bool = True) -> str:
    """
    Phase 1: stream the <think> block live in gray.
    Phase 2: stream the answer live as plain text, then snap to rendered markdown.
    """
    full_chunks = []
    answer_parts = []
    buffer = ""

    stream_iter = iter(stream)

    # Phase 1: consume think block, streaming it in gray
    for response in stream_iter:
        chunk = response.text
        full_chunks.append(chunk)
        buffer += chunk

        if "</think>" in buffer:
            end = buffer.index("</think>") + len("</think>")
            tail = buffer[:end]
            remainder = buffer[end:].lstrip("\n")

            if show_thinking:
                sys.stdout.write(GRAY + tail + RESET + "\n")
                sys.stdout.flush()

            buffer = ""
            if remainder:
                answer_parts.append(remainder)
            break
        else:
            safe = len(buffer)
            for i in range(1, len("</think>")):
                if buffer.endswith("</think>"[:i]):
                    safe = len(buffer) - i
                    break
            if show_thinking:
                sys.stdout.write(GRAY + buffer[:safe])
                sys.stdout.flush()
            buffer = buffer[safe:]

    # Phase 2: stream answer live as text, finalize as markdown
    with Live("", refresh_per_second=15, console=console) as live:
        for response in stream_iter:
            chunk = response.text
            full_chunks.append(chunk)
            answer_parts.append(chunk)
            live.update(Text("".join(answer_parts)))

        answer = "".join(answer_parts).strip()
        if answer:
            live.update(Markdown(answer))

    return "".join(full_chunks)


def load_history(path: str) -> list:
    with open(path) as f:
        data = json.load(f)
    # Support both old format (bare list) and new format (dict with messages key)
    messages = data["messages"] if isinstance(data, dict) else data
    for msg in messages:
        if msg["role"] == "assistant":
            msg["content"] = strip_thinking(msg["content"])
    return messages


def save_chat(history: list, path: Path, title: str, model: str) -> None:
    path.parent.mkdir(exist_ok=True)
    data = {
        "meta": {
            "title": title,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "model": model,
            "turns": len(history) // 2,
        },
        "messages": history,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def get_context(history: list, max_turns: int | None) -> list:
    if max_turns is None:
        return history
    return history[-(max_turns * 2):]


def prompt_save(generate_title_fn, history: list, resume: str | None, model: str) -> None:
    """Interactive save prompt: ask whether to save, suggest or enter a filename."""
    date_str = datetime.now().strftime("%Y-%m-%d")

    try:
        save = input("\nSave this chat? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return
    if save == "n":
        return

    # If resuming, offer to overwrite in-place
    if resume:
        try:
            overwrite = input(f"Overwrite original ({Path(resume).name})? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return
        if overwrite == "y":
            title = Path(resume).stem.split("_", 1)[-1]  # strip date prefix if present
            save_chat(history, Path(resume), title, model)
            print(f"Saved to {resume}")
            return

    # Suggest or enter a filename
    try:
        mode = input("Generate a title or enter your own? [G/e]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return

    if mode == "e":
        try:
            raw = input("Filename (without extension): ").strip()
        except (EOFError, KeyboardInterrupt):
            return
        slug = slugify(raw) or "chat"
    else:
        print("Generating title...", end="\r", flush=True)
        suggested = generate_title_fn(history)
        print(f"Suggested:  {date_str}_{suggested}.json")
        try:
            override = input("Press Enter to accept, or type a new name: ").strip()
        except (EOFError, KeyboardInterrupt):
            slug = suggested
        else:
            slug = slugify(override) if override else suggested

    path = CHATS_DIR / f"{date_str}_{slug}.json"
    save_chat(history, path, slug, model)
    print(f"Saved to {path}")


def chat_openai(
    model: str,
    base_url: str,
    resume: str | None = None,
    show_thinking: bool = True,
    max_turns: int | None = None,
):
    client = OpenAI(api_key="local", base_url=base_url)

    history = []
    if resume:
        history = load_history(resume)
        print(f"Resumed from {resume} ({len(history)} messages)\n")

    print("Nemotron Chat — type 'quit' to exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        context = get_context(history, max_turns)
        if max_turns and len(history) > len(context):
            dropped = (len(history) - len(context)) // 2
            print(f"  [context window: using last {max_turns} turns, {dropped} older turn(s) dropped]\n")

        print("Nemotron:")
        stream = client.chat.completions.create(
            model=model,
            messages=context,
            max_tokens=4096,
            stream=True,
        )
        full_response = stream_response_openai(stream, show_thinking=show_thinking)
        print()

        history.append({"role": "assistant", "content": full_response})

    if history:
        prompt_save(
            lambda h: generate_title_openai(client, model, h),
            history, resume, model,
        )


def chat_mlx(
    model: str,
    resume: str | None = None,
    show_thinking: bool = True,
    max_turns: int | None = None,
):
    from mlx_lm import load, stream_generate

    print("Loading model...")
    mlx_model, tokenizer = load(model, tokenizer_config={"trust_remote_code": True})

    history = []
    if resume:
        history = load_history(resume)
        print(f"Resumed from {resume} ({len(history)} messages)\n")

    print("Nemotron Chat — type 'quit' to exit\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        context = get_context(history, max_turns)
        if max_turns and len(history) > len(context):
            dropped = (len(history) - len(context)) // 2
            print(f"  [context window: using last {max_turns} turns, {dropped} older turn(s) dropped]\n")

        formatted = tokenizer.apply_chat_template(
            context, tokenize=False, add_generation_prompt=True
        )

        print("Nemotron:")
        stream = stream_generate(mlx_model, tokenizer, prompt=formatted, max_tokens=4096)
        full_response = stream_response_mlx(stream, show_thinking=show_thinking)
        print()

        history.append({"role": "assistant", "content": full_response})

    if history:
        prompt_save(
            lambda h: generate_title_mlx(mlx_model, tokenizer, h),
            history, resume, model,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", metavar="FILE", help="Resume from a prior chat JSON file")
    parser.add_argument("--no-thinking", action="store_true", help="Suppress thinking output")
    parser.add_argument("--max-turns", type=int, metavar="N", help="Sliding window: keep last N turns in context")
    parser.add_argument("--mlx", action="store_true", help="Use local MLX backend (Mac/Apple Silicon)")
    parser.add_argument("--model", metavar="MODEL", help="Model ID (overrides default for chosen backend)")
    parser.add_argument("--base-url", metavar="URL", help="OpenAI-compatible base URL (default: %(default)s)", default=DEFAULT_BASE_URL)
    args = parser.parse_args()

    if args.mlx:
        chat_mlx(
            model=args.model or DEFAULT_MODEL_MLX,
            resume=args.resume,
            show_thinking=not args.no_thinking,
            max_turns=args.max_turns,
        )
    else:
        chat_openai(
            model=args.model or DEFAULT_MODEL_OPENAI,
            base_url=args.base_url,
            resume=args.resume,
            show_thinking=not args.no_thinking,
            max_turns=args.max_turns,
        )
