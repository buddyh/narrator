"""Gemini image+text narration client."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple, List, Dict
import urllib.error
import urllib.request

from narrator.config import AppConfig

_API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/{model_path}:generateContent"
TEMPLATE_VERSION = "2026-01-31-02"


@dataclass
class UsageSummary:
    prompt_tokens: int = 0
    output_tokens: int = 0
    thoughts_tokens: int = 0
    total_tokens: int = 0
    attempts: int = 0
    prompt_text_tokens: int = 0
    prompt_image_tokens: int = 0


def generate_narration(
    png_bytes: bytes,
    app_name: str,
    window_title: str,
    config: AppConfig,
    history_lines: list[str],
    context_history: list[str],
    turn_index: int,
    verbose: bool = False,
    lane_overrides: Optional[Dict[str, Any]] = None,
) -> tuple[Optional[str], UsageSummary, str]:
    """Call Gemini to generate a sports-style narration line."""

    if not png_bytes:
        return None, UsageSummary(), "empty"

    best_line: Optional[str] = None
    usage_summary = UsageSummary()
    best_source = "none"
    char_target = (lane_overrides or {}).get("char_target")
    prompt = _build_prompt(
        app_name, window_title, config, history_lines, context_history, turn_index,
        char_target=char_target,
    )

    structured = _use_structured(config.gemini_model)
    for attempt in range(config.gemini_max_retries + 1):
        raw_text, info = _call_gemini(
            png_bytes,
            prompt,
            config,
            app_name=app_name,
            window_title=window_title,
            attempt=attempt,
            turn_index=turn_index,
            structured=structured,
            lane_overrides=lane_overrides,
        )
        _maybe_log_raw(config, raw_text, attempt, app_name, window_title)
        usage_summary = _accumulate_usage(usage_summary, info)
        parsed = _parse_structured_response(raw_text or "")
        if structured and (not raw_text or not parsed):
            structured = False
        if raw_text and not parsed:
            _maybe_log_trace(
                config,
                "gemini_parse_error",
                {"raw_text": raw_text},
                {
                    "attempt": attempt,
                    "app": app_name,
                    "window": window_title,
                    "turn": turn_index,
                },
            )
        observations = _normalize_list(parsed.get("observations"))
        actions = _normalize_list(parsed.get("actions"))
        observations = _sanitize_observations(observations)
        actions = _sanitize_actions(actions)
        topic = _topic_from_context(app_name, window_title, observations)
        allow_ui = not _is_known_app(app_name, window_title, topic)

        source = "commentary"
        line = _sanitize_line(parsed.get("commentary", "") or "")
        if not line and raw_text and not _looks_like_json(raw_text):
            line = _sanitize_line(raw_text)
            source = "raw"
        if line and _is_meta_response(line):
            line = None
            source = "filtered_meta"
        # UI-term filter disabled — was too aggressively killing valid Gemini
        # commentary for known apps (matching "menu", "icon", "tab", etc.)
        if not line:
            line = _compose_commentary(
                app_name,
                window_title,
                observations,
                actions,
                history_lines,
                context_history,
                turn_index,
                config.profanity_level,
                config.narration_style,
            )
            source = "composed"
        line = _strip_banned_opener(line, config.narration_style)
        line = _ensure_context(line, app_name, window_title, config.narration_style)
        line = _ensure_punctuation(line)
        line = _expand_line(
            line,
            config,
            app_name,
            window_title,
            observations,
            actions,
            history_lines,
            context_history,
            turn_index,
        )
        if line:
            best_line = line
            best_source = source

        if verbose:
            _log_response(info, raw_text or "", line, attempt, source)
            if observations:
                print(f"[Gemini] Observations: {observations}")
            if actions:
                print(f"[Gemini] Actions: {actions}")

        if _meets_constraints(line, config, app_name, window_title, lane_overrides=lane_overrides):
            return line, usage_summary, source

        if verbose and config.debug_constraints:
            print(
                "[Gemini] Constraints not met: "
                f"{_constraint_report(line, config, app_name, window_title, lane_overrides=lane_overrides)}"
            )

        prompt = _build_retry_prompt(
            app_name,
            window_title,
            config,
            history_lines,
            context_history,
            turn_index,
            line,
            char_target=char_target,
        )

    if best_line:
        return best_line, usage_summary, best_source
    return None, usage_summary, "none"


def _call_gemini(
    png_bytes: bytes,
    prompt: str,
    config: AppConfig,
    app_name: str,
    window_title: str,
    attempt: int,
    turn_index: int,
    structured: bool,
    lane_overrides: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[str], dict[str, Any]]:
    payload = _build_payload(png_bytes, prompt, config, structured, lane_overrides=lane_overrides)
    model_path = config.gemini_model.strip()
    if not model_path.startswith("models/"):
        model_path = f"models/{model_path}"

    url = _API_URL_TEMPLATE.format(model_path=model_path)
    safe_url = _redact_api_key(url)
    image_sha = hashlib.sha1(png_bytes).hexdigest()[:12]
    _maybe_log_trace(
        config,
        "gemini_request",
        payload,
        {
            "attempt": attempt,
            "app": app_name,
            "window": window_title,
            "turn": turn_index,
            "url": safe_url,
            "image_bytes": len(png_bytes),
            "image_sha": image_sha,
            "structured": structured,
        },
    )

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": config.gemini_api_key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=config.gemini_timeout_sec) as resp:
            status = getattr(resp, "status", None)
            response_body = resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        _maybe_log_trace(
            config,
            "gemini_error_response",
            {"error": detail},
            {
                "attempt": attempt,
                "app": app_name,
                "window": window_title,
                "turn": turn_index,
                "status": exc.code,
                "url": safe_url,
                "image_bytes": len(png_bytes),
                "image_sha": image_sha,
            },
        )
        raise RuntimeError(f"Gemini API error ({exc.code}): {detail}") from exc
    except urllib.error.URLError as exc:
        _maybe_log_trace(
            config,
            "gemini_error_connection",
            {"error": str(exc)},
            {
                "attempt": attempt,
                "app": app_name,
                "window": window_title,
                "turn": turn_index,
                "url": safe_url,
                "image_bytes": len(png_bytes),
                "image_sha": image_sha,
            },
        )
        raise RuntimeError(f"Gemini API connection error: {exc}") from exc

    response_text = response_body.decode("utf-8", errors="replace")
    try:
        response_json = json.loads(response_text)
    except json.JSONDecodeError as exc:
        _maybe_log_trace(
            config,
            "gemini_response_decode_error",
            {"raw": response_text},
            {
                "attempt": attempt,
                "app": app_name,
                "window": window_title,
                "turn": turn_index,
                "status": status,
                "url": safe_url,
                "error": str(exc),
                "image_bytes": len(png_bytes),
                "image_sha": image_sha,
            },
        )
        raise RuntimeError(f"Gemini API invalid JSON response: {exc}") from exc
    _maybe_log_trace(
        config,
        "gemini_response",
        response_json,
        {
            "attempt": attempt,
            "app": app_name,
            "window": window_title,
            "turn": turn_index,
            "status": status,
            "url": safe_url,
            "image_bytes": len(png_bytes),
            "image_sha": image_sha,
        },
    )
    return _extract_text(response_json)


def _redact_api_key(url: str) -> str:
    return re.sub(r"(key=)[^&]+", r"\1***", url)


def _use_structured(model_name: str) -> bool:
    lowered = (model_name or "").lower()
    if "gemini-3-flash-preview" in lowered:
        return False
    return True


def _maybe_log_trace(
    config: AppConfig,
    event: str,
    payload: dict[str, Any] | list[Any] | str,
    meta: dict[str, Any],
) -> None:
    if not config.gemini_trace_log:
        return
    ts = time.time()
    header = {"ts": f"{ts:.3f}", "event": event, **meta}
    try:
        with open(config.gemini_trace_log, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(header, ensure_ascii=True) + "\n")
            if isinstance(payload, str):
                handle.write(payload + "\n")
            else:
                redacted = _redact_large_blobs(payload)
                handle.write(json.dumps(redacted, ensure_ascii=True) + "\n")
    except OSError:
        return


def _redact_large_blobs(payload: Any) -> Any:
    if isinstance(payload, dict):
        redacted: dict[str, Any] = {}
        for key, value in payload.items():
            if key == "data" and isinstance(value, str) and len(value) > 128:
                redacted[key] = f"<redacted {len(value)} chars>"
            else:
                redacted[key] = _redact_large_blobs(value)
        return redacted
    if isinstance(payload, list):
        return [_redact_large_blobs(item) for item in payload]
    return payload


def _style_header(config: AppConfig, char_target: Optional[str] = None) -> str:
    """Return the style-specific system prompt header."""
    ct = char_target or "90-150"
    json_footer = (
        "- Never preface with phrases like \"here is the JSON\" or wrap output in code fences.\n"
        "- Output must be a raw JSON object only.\n"
        "- If the app is well-known (e.g., X, Chrome, Safari), avoid UI element descriptions; focus on activity + content.\n"
        "- If the app is obscure/unknown, include UI details in the observations and commentary.\n"
        "- No quotes in the commentary.\n"
        "- observations: 3-6 short phrases naming visible UI elements, labels, or layout details.\n"
        "- actions: 1-3 short verbs/phrases (scrolling, typing, switching tabs).\n"
        "- Do not include passwords, emails, phone numbers, or account numbers.\n"
        "- No markdown, no extra keys.\n\n"
        "JSON format:\n"
        "{\n"
        "  \"commentary\": \"...\",\n"
        "  \"observations\": [\"...\", \"...\"],\n"
        "  \"actions\": [\"...\"]\n"
        "}\n\n"
    )
    style = config.narration_style
    profanity = config.profanity_level

    if style == "nature":
        return (
            "You are David Attenborough narrating a nature documentary about a human using their computer. "
            "Return ONLY JSON.\n\n"
            "Rules:\n"
            f"- commentary: 1-2 sentences, {ct} chars total. Keep it tight.\n"
            "- Must mention the app name or window title.\n"
            "- Include at least two concrete details from the screen.\n"
            "- Tone: hushed, reverent, wondrous, like observing a rare creature in its habitat.\n"
            "- Style: treat every mouse click and scroll as a fascinating behavioral ritual. "
            "The user is a specimen. Their desktop is the savanna. Their apps are the ecosystem.\n"
            "- Narrate in third person as 'the human', 'our subject', 'the creature'.\n"
            "- Find beauty and drama in the mundane. A tab switch is a migration. A scroll is foraging.\n"
            "- Gentle wit, dry humor, understated observations. Never mean.\n"
            f"- Profanity level: {profanity} (keep it classy even at high).\n"
            "- Avoid repeating phrasing from recent commentary.\n"
            + json_footer
        )

    if style == "horror":
        return (
            "You are a horror movie narrator watching someone use their computer. Something is wrong. "
            "Return ONLY JSON.\n\n"
            "Rules:\n"
            f"- commentary: 1-2 sentences, {ct} chars total. Keep it tight.\n"
            "- Must mention the app name or window title.\n"
            "- Include at least two concrete details from the screen.\n"
            "- Tone: creeping dread, ominous foreshadowing, slow-burn suspense.\n"
            "- Style: treat every action as a warning sign. The cursor moves too slowly. The tab count is wrong. "
            "Something in the notification bar does not belong. Every mundane action hides menace.\n"
            "- Narrate in third person. The user does not know what is coming.\n"
            "- Build tension from nothing. A loading spinner is a countdown. A new email is a threat.\n"
            "- Whispered urgency, not gore. Psychological, not graphic.\n"
            f"- Profanity level: {profanity} (dread over shock value).\n"
            "- Avoid repeating phrasing from recent commentary.\n"
            + json_footer
        )

    if style == "noir":
        return (
            "You are a hard-boiled noir detective narrating a case that happens to involve someone at a computer. "
            "Return ONLY JSON.\n\n"
            "Rules:\n"
            f"- commentary: 1-2 sentences, {ct} chars total. Keep it tight.\n"
            "- Must mention the app name or window title.\n"
            "- Include at least two concrete details from the screen.\n"
            "- Tone: world-weary, cynical, poetic. Raymond Chandler meets the desktop.\n"
            "- Style: first person. You are the detective, the screen is the crime scene. "
            "Every tab is a lead. Every search is a dead end. The cursor is the only witness.\n"
            "- Use metaphor freely. The browser is a dame with secrets. The terminal is a dark alley.\n"
            "- Dry, bitter wit. Nothing surprises you anymore.\n"
            f"- Profanity level: {profanity} (hardboiled, not vulgar).\n"
            "- Avoid repeating phrasing from recent commentary.\n"
            + json_footer
        )

    if style == "reality_tv":
        return (
            "You are a ditzy reality TV contestant in the confessional booth, commenting on what you just saw someone doing on their computer. "
            "You don't really understand technology but you have strong opinions anyway. "
            "Return ONLY JSON.\n\n"
            "Rules:\n"
            f"- commentary: 1-2 sentences, {ct} chars total. Keep it tight.\n"
            "- Reference the app or website naturally but dumb it down or get it slightly wrong. "
            "Like calling a terminal 'that hacker screen thing' or VS Code 'some kind of coding situation.' "
            "Slack is 'the group chat.' A spreadsheet is 'the number grid thing.' "
            "You can say Chrome, Google, Safari, etc. Just don't be formal about it.\n"
            "- Include at least one concrete detail from the screen but misinterpret or oversimplify it.\n"
            "- Tone: bubbly, confused, judgmental. You're trying your best but you're out of your depth. "
            "Talk like you're explaining it to your friend at brunch.\n"
            "- Use filler words naturally (like, honestly, literally, I don't know, wait). "
            "Trail off sometimes. Jump to wrong conclusions confidently.\n"
            "- React dramatically to mundane things. A git commit is 'deleting everything.' "
            "Multiple tabs means 'they're clearly going through something.'\n"
            f"- Profanity level: {profanity} (reality TV sass).\n"
            "- Avoid repeating phrasing from recent commentary.\n"
            + json_footer
        )

    if style == "asmr":
        return (
            "You are a soothing ASMR narrator whispering about someone using their computer. "
            "Return ONLY JSON.\n\n"
            "Rules:\n"
            f"- commentary: 1-2 sentences, {ct} chars total. Keep it tight.\n"
            "- Must mention the app name or window title.\n"
            "- Include at least two concrete details from the screen.\n"
            "- Tone: gentle, hypnotic, calming. Every word is a warm blanket.\n"
            "- Style: describe actions in slow, sensory detail. The click of the trackpad. "
            "The soft glow of the screen. The quiet rhythm of the scroll. Make it cozy.\n"
            "- Narrate in second person. Guide the listener through the scene like a meditation.\n"
            "- The humor comes from the contrast between the soothing tone and the mundane chaos on screen.\n"
            f"- Profanity level: {profanity} (whispered, never harsh).\n"
            "- Avoid repeating phrasing from recent commentary.\n"
            + json_footer
        )

    if style == "wrestling":
        return (
            "You are a professional wrestling announcer calling the action on someone's computer screen. "
            "Return ONLY JSON.\n\n"
            "Rules:\n"
            f"- commentary: 1-2 sentences, {ct} chars total. Keep it tight.\n"
            "- Must mention the app name or window title.\n"
            "- Include at least two concrete details from the screen.\n"
            "- Tone: maximum hype, over-the-top excitement, disbelief at every action.\n"
            "- Style: treat every click as a finishing move. Every tab switch is a betrayal. "
            "Every app open is a new challenger entering the ring. The cursor is the champion.\n"
            "- Use wrestling language: OH MY GOD, BAH GAWD, SOMEBODY STOP THE MATCH, "
            "FROM THE TOP ROPE, THAT MAN HAD A FAMILY.\n"
            "- Alternate between play-by-play and color commentary. Sell every moment.\n"
            f"- Profanity level: {profanity} (PG-13 wrestling broadcast).\n"
            "- Avoid repeating phrasing from recent commentary.\n"
            + json_footer
        )

    # Default: sports style
    return (
        "You are analyzing a screenshot and app context to produce a sports-style narration. "
        "Return ONLY JSON.\n\n"
        "Rules:\n"
        f"- commentary: 1-2 punchy sentences, {ct} chars total. Keep it tight and fast.\n"
        "- Must mention the app name or window title.\n"
        "- Include at least two concrete details from the screen (content cues for known apps; UI details for unknown apps).\n"
        "- Tone: sports announcer, witty, continuous, present tense with a funny edge.\n"
        "- Style: playful, a little deprecating, colorful language, light roast of the user's dithering.\n"
        "- Use casual, punchy profanity where it fits (no slurs, no harassment).\n"
        f"- Profanity level: {profanity} (no slurs, no harassment).\n"
        "- Avoid openers like \"Our user\" or \"Ladies and gentlemen\".\n"
        "- Avoid repeating phrasing from recent commentary.\n"
        + json_footer
    )


def _build_prompt(
    app_name: str,
    window_title: str,
    config: AppConfig,
    history_lines: list[str],
    context_history: list[str],
    turn_index: int,
    char_target: Optional[str] = None,
) -> str:
    safe_app = app_name or "Unknown App"
    safe_title = _clean_window_title(window_title or "")

    header = _style_header(config, char_target=char_target)

    context_block = f"Turn: {turn_index}\nApp: {safe_app}\nWindow: {safe_title}"

    history_block = ""
    if history_lines:
        trimmed = history_lines[-config.gemini_history_lines :]
        history_lines_text = "\n".join(f"- {line}" for line in trimmed)
        history_block = (
            "\n\nRecent commentary (continue the narrative thread, build on what came before, never repeat):\n"
            f"{history_lines_text}"
        )

    context_history_block = ""
    if context_history:
        trimmed_ctx = context_history[-config.gemini_history_contexts :]
        context_lines = "\n".join(f"- {line}" for line in trimmed_ctx)
        context_history_block = (
            "\n\nRecent context snapshots:\n"
            f"{context_lines}"
        )

    return f"{header}\n{context_block}{context_history_block}{history_block}"


def _build_retry_prompt(
    app_name: str,
    window_title: str,
    config: AppConfig,
    history_lines: list[str],
    context_history: list[str],
    turn_index: int,
    last_line: Optional[str],
    char_target: Optional[str] = None,
) -> str:
    base = _build_prompt(
        app_name, window_title, config, history_lines, context_history, turn_index,
        char_target=char_target,
    )
    if not last_line:
        return base + "\n\nYour last JSON was missing details. Add more specific observations and a longer commentary."
    return (
        base
        + "\n\nYour last JSON was too generic or too short. Add more specific observations and a longer commentary.\n"
        f"Previous: {last_line}"
    )


def _build_payload(
    png_bytes: bytes, prompt: str, config: AppConfig, structured: bool,
    lane_overrides: Optional[Dict[str, Any]] = None,
) -> dict[str, Any]:
    image_b64 = base64.b64encode(png_bytes).decode("ascii")

    thinking_budget = (lane_overrides or {}).get("thinking_budget", 128)
    generation_config: dict[str, Any] = {
        "temperature": config.gemini_temperature,
        "maxOutputTokens": config.gemini_max_output_tokens,
        "thinkingConfig": {"thinkingBudget": thinking_budget},
    }
    if structured:
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseSchema"] = {
            "type": "object",
            "properties": {
                "commentary": {"type": "string"},
                "observations": {"type": "array", "items": {"type": "string"}},
                "actions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["commentary", "observations", "actions"],
        }

    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_b64,
                        }
                    },
                    {"text": prompt},
                ],
            }
        ],
        "generationConfig": generation_config,
    }


def _extract_text(response_json: dict[str, Any]) -> tuple[Optional[str], dict[str, Any]]:
    info: dict[str, Any] = {}
    prompt_feedback = response_json.get("promptFeedback")
    if prompt_feedback and prompt_feedback.get("blockReason"):
        info["blocked"] = prompt_feedback.get("blockReason")
        return None, info

    candidates = response_json.get("candidates") or []
    info["candidate_count"] = len(candidates)
    usage = response_json.get("usageMetadata")
    if usage:
        info["usage"] = usage
    for candidate in candidates:
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        for part in parts:
            text = part.get("text")
            if text:
                return text, info
    return None, info


def _accumulate_usage(summary: UsageSummary, info: dict[str, Any]) -> UsageSummary:
    usage = info.get("usage") or {}
    prompt_tokens = int(usage.get("promptTokenCount") or 0)
    candidate_tokens = int(usage.get("candidatesTokenCount") or 0)
    thoughts_tokens = int(usage.get("thoughtsTokenCount") or 0)
    total_tokens = int(usage.get("totalTokenCount") or 0)
    details = usage.get("promptTokensDetails") or []
    text_tokens = 0
    image_tokens = 0
    for detail in details:
        modality = str(detail.get("modality") or "").upper()
        count = int(detail.get("tokenCount") or 0)
        if modality == "TEXT":
            text_tokens += count
        elif modality == "IMAGE":
            image_tokens += count
    summary.prompt_tokens += prompt_tokens
    summary.output_tokens += candidate_tokens + thoughts_tokens
    summary.thoughts_tokens += thoughts_tokens
    summary.total_tokens += total_tokens or (prompt_tokens + candidate_tokens + thoughts_tokens)
    summary.prompt_text_tokens += text_tokens
    summary.prompt_image_tokens += image_tokens
    summary.attempts += 1
    return summary


def _maybe_log_raw(
    config: AppConfig,
    raw_text: Optional[str],
    attempt: int,
    app_name: str,
    window_title: str,
) -> None:
    if not config.gemini_raw_log:
        return
    ts = time.time()
    header = (
        f"\n--- ts={ts:.3f} attempt={attempt} app={app_name} window={window_title}\n"
    )
    try:
        with open(config.gemini_raw_log, "a", encoding="utf-8") as handle:
            handle.write(header)
            handle.write(raw_text or "<empty>")
            handle.write("\n")
    except OSError:
        return


def _sanitize_line(text: str) -> Optional[str]:
    if not text:
        return None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    line = " ".join(lines)
    line = line.replace("**", "").replace("*", "").strip()
    line = re.sub(r"^[{\\[]\\s*commentary\\s*[:=]\\s*", "", line, flags=re.IGNORECASE)
    line = re.sub(r"^commentary\\s*[:=]\\s*", "", line, flags=re.IGNORECASE)
    line = line.strip("{}[] ")
    line = line.replace("\u201c", "").replace("\u201d", "").replace("\u2019", "'")
    line = line.replace("\"", "").replace("`", "")
    line = _ascii_only(line)
    if len(line) <= 5:
        return None
    if len(line) > 520:
        return line[:517].rsplit(" ", 1)[0] + "..."
    return line


def _count_sentences(text: str) -> int:
    if not text:
        return 0
    parts = re.split(r"[.!?]+", text)
    return sum(1 for part in parts if part.strip())


def _count_words(text: str) -> int:
    return len([word for word in text.split() if word.strip()])


def _looks_like_json(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("{") and stripped.endswith("}")


def _meets_constraints(
    line: Optional[str], config: AppConfig, app_name: str, window_title: str,
    lane_overrides: Optional[Dict[str, Any]] = None,
) -> bool:
    if not line:
        return False
    min_chars = (lane_overrides or {}).get("min_chars", config.gemini_min_chars)
    min_words = (lane_overrides or {}).get("min_words", config.gemini_min_words)
    if len(line) < min_chars:
        return False
    if _count_words(line) < min_words:
        return False
    if _count_sentences(line) < config.gemini_min_sentences:
        return False
    if not _ends_with_punctuation(line):
        return False
    if _has_banned_opener(line, config.narration_style):
        return False
    return True


def _constraint_report(
    line: Optional[str], config: AppConfig, app_name: str, window_title: str,
    lane_overrides: Optional[Dict[str, Any]] = None,
) -> str:
    if not line:
        return "empty"
    min_chars = (lane_overrides or {}).get("min_chars", config.gemini_min_chars)
    min_words = (lane_overrides or {}).get("min_words", config.gemini_min_words)
    return (
        f"chars={len(line)}/{min_chars} "
        f"words={_count_words(line)}/{min_words} "
        f"sentences={_count_sentences(line)}/{config.gemini_min_sentences} "
        f"ends={_ends_with_punctuation(line)} "
        f"banned={_has_banned_opener(line, config.narration_style)} "
        f"context={'skip' if config.narration_style in _SKIP_FORMAL_CONTEXT_STYLES else _mentions_context(line, app_name, window_title)}"
    )


def _log_response(
    info: dict[str, Any],
    raw_text: str,
    line: Optional[str],
    attempt: int,
    source: str,
) -> None:
    usage = info.get("usage")
    candidates = info.get("candidate_count")
    blocked = info.get("blocked")
    debug = f"candidates={candidates} attempt={attempt}"
    if usage:
        debug = f"{debug} usage={usage}"
    if blocked:
        debug = f"{debug} blocked={blocked}"
    print(f"[Gemini] Response: {debug}")
    if raw_text:
        snippet = raw_text.replace("\n", " ")
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        print(f"[Gemini] Raw: {snippet}")
    if line:
        print(f"[Gemini] Line source={source} chars={len(line)}")


def _ends_with_punctuation(text: str) -> bool:
    return text.rstrip().endswith((".", "!", "?"))


def _banned_openers(style: str) -> tuple[str, ...]:
    common = (
        "buddy is",
        "buddy's",
        "ladies and gentlemen",
    )
    if style == "nature":
        return common
    if style == "noir":
        return common + ("the user",)
    if style == "asmr":
        return common + ("our user",)
    if style == "wrestling":
        return common + ("our user",)
    if style in ("horror", "reality_tv"):
        return common
    return common + (
        "our user",
        "our player",
        "our heavy hitter",
        "our digital athlete",
        "our listener",
    )


def _has_banned_opener(text: str, style: str = "sports") -> bool:
    lowered = text.strip().lower()
    return any(lowered.startswith(prefix) for prefix in _banned_openers(style))


def _strip_banned_opener(text: Optional[str], style: str = "sports") -> Optional[str]:
    if not text:
        return text
    lowered = text.strip().lower()
    for prefix in _banned_openers(style):
        if lowered.startswith(prefix):
            trimmed = text.strip()[len(prefix) :].lstrip(" ,:-")
            return trimmed or text
    return text


def _mentions_context(text: str, app_name: str, window_title: str) -> bool:
    lowered = text.lower()
    app_tokens = re.findall(r"[A-Za-z0-9]+", app_name.lower())
    window_tokens = re.findall(r"[A-Za-z0-9]+", window_title.lower())
    tokens = [t for t in app_tokens + window_tokens if len(t) >= 4]
    if not tokens:
        return True
    return any(token in lowered for token in tokens[:8])


def _parse_structured_response(raw_text: str) -> Dict[str, Any]:
    if not raw_text:
        return {}
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _normalize_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return []


def _sanitize_observations(items: List[str]) -> List[str]:
    cleaned: List[str] = []
    for item in items:
        text = re.sub(r"\s+", " ", item).strip()
        text = re.sub(r"\b\d{5,}\b", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        if len(text) < 4:
            continue
        if len(text) > 80:
            text = text[:77].rsplit(" ", 1)[0] + "..."
        cleaned.append(_ascii_only(text))
    return cleaned[:6]


def _sanitize_actions(items: List[str]) -> List[str]:
    cleaned: List[str] = []
    for item in items:
        text = re.sub(r"\s+", " ", item).strip()
        text = text.rstrip(".")
        if len(text) < 3:
            continue
        if len(text) > 40:
            text = text[:37].rsplit(" ", 1)[0] + "..."
        cleaned.append(_ascii_only(text))
    return cleaned[:3]


def _token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z]{4,}", text.lower()))


def _too_similar(line: str, history_lines: List[str]) -> bool:
    if not history_lines:
        return False
    tokens = _token_set(line)
    if not tokens:
        return False
    for prev in history_lines[-3:]:
        if not prev:
            continue
        prev_lower = prev.lower()
        lower = line.lower()
        if lower in prev_lower or prev_lower in lower:
            return True
        prev_tokens = _token_set(prev)
        if not prev_tokens:
            continue
        jaccard = len(tokens & prev_tokens) / len(tokens | prev_tokens)
        if jaccard >= 0.5:
            return True
    return False


def _select_variant(options: List[str], history_lines: List[str], turn_index: int) -> str:
    if not options:
        return ""
    recent = history_lines[-4:]
    if not recent:
        return options[turn_index % len(options)]

    scored: List[tuple[float, int]] = []
    for idx, candidate in enumerate(options):
        candidate_lower = candidate.lower()
        cand_tokens = _token_set(candidate_lower)
        if not cand_tokens:
            scored.append((0.0, idx))
            continue
        max_overlap = 0.0
        for prev in recent:
            prev_lower = prev.lower()
            if candidate_lower in prev_lower or prev_lower in candidate_lower:
                max_overlap = 1.0
                break
            prev_tokens = _token_set(prev_lower)
            if not prev_tokens:
                continue
            overlap = len(cand_tokens & prev_tokens) / len(cand_tokens | prev_tokens)
            if overlap > max_overlap:
                max_overlap = overlap
        scored.append((max_overlap, idx))

    min_overlap = min(score for score, _ in scored)
    best = [idx for score, idx in scored if score == min_overlap]
    pick = best[turn_index % len(best)]
    return options[pick]


def _has_ui_terms(text: str) -> bool:
    lowered = text.lower()
    patterns = [
        r"\bsidebar\b",
        r"\btoolbar\b",
        r"\btabs?\b",
        r"\baddress bar\b",
        r"\bpanel\b",
        r"\bpane\b",
        r"\bbutton\b",
        r"\bmenu\b",
        r"\bicon\b",
        r"\bnavigation\b",
        r"\bnav\b",
        r"\btitle bar\b",
        r"\bscrollbar\b",
        r"\bstatus bar\b",
    ]
    return any(re.search(pattern, lowered) for pattern in patterns)


def _is_meta_response(text: str) -> bool:
    lowered = text.lower()
    meta_phrases = [
        "here is the json",
        "here's the json",
        "json requested",
        "as requested",
    ]
    if any(phrase in lowered for phrase in meta_phrases):
        return True
    if "```json" in lowered or "```" in lowered:
        return True
    return False


def generate_context_narration(
    app_name: str,
    window_title: str,
    config: AppConfig,
    history_lines: list[str],
    context_history: list[str],
    turn_index: int,
) -> Optional[str]:
    """Generate a fast, context-only narration line without image analysis."""

    line = _compose_commentary(
        app_name,
        window_title,
        [],
        [],
        history_lines,
        context_history,
        turn_index,
        config.profanity_level,
        config.narration_style,
    )
    line = _strip_banned_opener(line, config.narration_style)
    line = _ensure_context(line, app_name, window_title, config.narration_style)
    line = _ensure_punctuation(line)
    line = _expand_line(
        line,
        config,
        app_name,
        window_title,
        [],
        [],
        history_lines,
        context_history,
        turn_index,
    )
    return line


def _topic_from_context(app_name: str, window_title: str, observations: List[str]) -> str:
    text = f"{app_name} {window_title} {' '.join(observations)}".lower()
    if "x.com" in text or "twitter" in text or re.search(r"\bx\b", window_title.lower()):
        return "x"
    if "terminal" in text or "tmux" in text or "iterm" in text:
        return "terminal"
    if "spotify" in text or "music" in text:
        return "music"
    if "chrome" in text or "safari" in text or "browser" in text:
        return "browser"
    if "slack" in text or "discord" in text or "messages" in text:
        return "chat"
    return "generic"


def _is_known_app(app_name: str, window_title: str, topic: str) -> bool:
    if topic in {"x", "browser", "chat", "music", "terminal"}:
        return True
    text = f"{app_name} {window_title}".lower()
    known = [
        "chrome",
        "safari",
        "firefox",
        "brave",
        "arc",
        "edge",
        "x",
        "twitter",
        "slack",
        "discord",
        "terminal",
        "iterm",
        "spotify",
        "messages",
        "outlook",
        "teams",
        "zoom",
        "notion",
        "figma",
        "finder",
        "mail",
        "chatgpt",
        "claude",
        "cursor",
        "vscode",
        "code",
    ]
    return any(token in text for token in known)


def _build_observation_pool(
    observations: List[str], topic: str, allow_ui: bool
) -> List[str]:
    if not allow_ui:
        return []
    if observations:
        return observations
    if topic == "x":
        return [
            "a vertical feed of posts",
            "a left navigation rail",
            "engagement icons under each post",
            "a compose area ready to go",
        ]
    if topic == "terminal":
        return [
            "a grid of monospace text",
            "pane borders splitting the window",
            "prompt lines stacked in rows",
            "a blinking cursor on standby",
        ]
    if topic == "music":
        return [
            "a track list stacked in rows",
            "playback controls along the bottom",
            "album art in a side panel",
            "a progress bar creeping forward",
        ]
    if topic == "browser":
        return [
            "a tab strip up top",
            "the address bar and toolbar icons",
            "a scrolling content column",
            "a sidebar or secondary panel",
        ]
    if topic == "chat":
        return [
            "a channel list on the left",
            "a message thread in the center",
            "a composer bar at the bottom",
            "avatars lined down the side",
        ]
    return [
        "stacked panels across the window",
        "a toolbar row with controls",
        "dense text blocks in the main pane",
        "small buttons along the top bar",
    ]


def _build_action_pool(
    actions: List[str], topic: str, profanity_level: str
) -> List[str]:
    if actions:
        return actions
    if topic == "x":
        pool = [
            "doomscrolling the feed",
            "scrolling the timeline",
            "hovering to post",
            "refreshing the feed for no good reason",
        ]
        if profanity_level in {"low", "high"}:
            pool.append("doomscrolling X and wasting time")
        if profanity_level == "high":
            pool.append("doomscrolling X and wasting a bunch of fucking time")
        return pool
    if topic == "terminal":
        return ["typing commands", "switching panes", "running a script"]
    if topic == "music":
        return ["browsing tracks", "queueing a song", "skipping around the playlist"]
    if topic == "browser":
        pool = ["scrolling a page", "tab hopping", "digging through a site"]
        if profanity_level == "high":
            pool.append("scrolling like the answer will magically appear")
        return pool
    if topic == "chat":
        return ["reading messages", "drafting a reply", "skimming the thread"]
    return ["moving through the interface", "browsing around", "poking at controls"]


def _action_clause(action: str, topic: str, history_lines: List[str], turn_index: int) -> str:
    base = action or "moving through the interface"
    clauses = [
        f"{base} with no sign of slowing",
        f"{base} like the clock is ticking",
        f"{base} and keeping the tempo up",
        f"{base} with the cursor stalking the next click",
        f"{base} with zero hesitation",
        f"{base} and letting the minutes leak",
        f"{base} like its the main event",
        f"{base} while everything else waits",
        f"{base} with the pace locked in",
        f"{base} and dragging the clock with it",
        f"{base} in full control of the lane",
        f"{base} and refusing to slow down",
    ]
    if topic == "x":
        clauses.extend(
            [
                f"{base} like a full-time habit",
                f"{base} like its paid hourly",
                f"{base} like the feed is paying rent",
                f"{base} and feeding the algorithm",
            ]
        )
    if topic == "terminal":
        clauses.extend(
            [
                f"{base} like its muscle memory",
                f"{base} with the prompt calling the shots",
            ]
        )
    return _select_variant(clauses, history_lines, turn_index)


def _window_part(window: str) -> str:
    if not window:
        return ""
    return f" on {window}"


def _content_cue(app_name: str, window_title: str, topic: str) -> str:
    window = _safe_snippet(window_title)
    if topic == "x":
        return "the X timeline"
    if topic == "browser":
        if window:
            return f"the page {window}"
        return "the current page"
    if topic == "terminal":
        return "the command line"
    if topic == "chat":
        return "the conversation thread"
    if topic == "music":
        return "the track list"
    if window:
        return f"the screen labeled {window}"
    if app_name:
        return f"{app_name}"
    return "the screen"


def _flair_pool(topic: str, profanity_level: str, style: str = "sports") -> List[str]:
    if style in _STYLE_TEMPLATES:
        return _STYLE_TEMPLATES[style]["closes"]
    if style == "nature":
        return [
            "The creature pauses, hovering, as if sensing something just beyond the fold.",
            "And so the scroll continues, an ancient foraging instinct dressed in modern glass.",
            "Each scroll reveals new territory, yet the creature presses on, unsatisfied.",
            "There is a patience here, a stillness that belies the chaos of the screen.",
        ]
    base = [
        "No shortcuts, just patience and a cursor with something to prove.",
        "This routine eats minutes for breakfast and asks for seconds.",
        "All rhythm and restraint, like they are waiting for the perfect click.",
        "The tempo says focus, the cursor says chaos.",
        "Its the slow burn of attention and the fast burn of time.",
        "This is focus theater with the clock as the audience.",
        "Time is the opponent and it is winning.",
        "You can feel the hesitation in every pause.",
        "The cursor is hovering like a referee about to blow a whistle.",
        "This is a long drive with no touchdown in sight.",
    ]
    if profanity_level in {"low", "high"}:
        base.extend(
            [
                "This stretch is chewing up time like its a full-time job.",
                "The clock is bleeding and nobody is stopping it.",
                "This is time leakage on a professional level.",
                "Minutes are dropping like loose change.",
            ]
        )
    if profanity_level == "high":
        base.extend(
            [
                "This is a slow-motion time leak and it knows it.",
                "All this motion just to waste a bunch of fucking time.",
                "The clock is getting robbed in broad daylight.",
                "This is procrastination in a jersey.",
            ]
        )
    if topic == "x":
        extra = [
            "Over on X, the doomscroll is doing cardio.",
            "That feed is a damn gravity well, and the scroll wheel knows it.",
            "X is open and the timeline is undefeated right now.",
            "The feed is loud and the discipline is quiet.",
            "The timeline is pulling like gravity and its not letting go.",
        ]
        if profanity_level in {"low", "high"}:
            extra.append("Just burning time in the feed like its a sport.")
        if profanity_level == "high":
            extra.append("Just wasting a bunch of fucking time doomscrolling X.")
        return base + extra
    if topic == "terminal":
        extra = [
            "Terminal energy all day, no GUI safety net in sight.",
            "Command line cooking, low heat and constant stirring.",
            "This is keys and prompts, nothing else matters.",
            "Straight command line grind with no shortcuts.",
        ]
        if profanity_level == "high":
            extra.append("Its pure terminal grit, no bullshit, just the prompt.")
        return base + extra
    if topic == "browser":
        extra = [
            "Tabs everywhere, confidence somewhere.",
            "One more scroll, then another, then another.",
            "This page is a treadmill and the feet keep moving.",
            "Its endless browsing disguised as progress.",
        ]
        if profanity_level in {"low", "high"}:
            extra.append("This page is a sinkhole for minutes.")
        if profanity_level == "high":
            extra.append("Chrome is open and the procrastination is loud.")
        return base + extra
    return base


def _compose_nature_commentary(
    app_name: str,
    window_title: str,
    observations: List[str],
    actions: List[str],
    history_lines: List[str],
    context_history: List[str],
    turn_index: int,
) -> str:
    """Compose Attenborough-style nature documentary fallback narration."""
    app = app_name or "this application"
    window = _safe_snippet(window_title)
    topic = _topic_from_context(app_name, window_title, observations)
    content_cue = _content_cue(app_name, window_title, topic)

    transition = ""
    if context_history and len(context_history) >= 2:
        prev_app = context_history[-2].split("|", 1)[0].strip()
        if prev_app and prev_app != app_name:
            migration_templates = [
                "Having departed the familiar grounds of {prev_app}, ",
                "The migration from {prev_app} is complete. ",
                "Leaving {prev_app} behind, ",
                "With a decisive click, the creature abandons {prev_app}. ",
                "The territory of {prev_app} grows distant now. ",
            ]
            transition = _select_variant(
                [t.format(prev_app=prev_app) for t in migration_templates],
                history_lines, turn_index,
            )

    lead_templates = [
        "Here, in the digital wilderness of {app}, the human settles into {content_cue}.",
        "And now we observe the creature navigating {app}, drawn to {content_cue}.",
        "The subject has found its way to {app}, where {content_cue} awaits.",
        "Quietly, almost imperceptibly, the human turns to {app} and {content_cue}.",
        "In the glow of {app}, the creature fixates on {content_cue}.",
        "Watch closely now. The human approaches {app}, seeking {content_cue}.",
        "The creature stirs. {app} has captured its attention, {content_cue} in particular.",
        "Remarkable. The human has chosen {app}, eyes locked on {content_cue}.",
    ]
    detail_templates = [
        "The fingers move with quiet purpose, a ritual performed a thousand times before.",
        "Each scroll reveals new territory, yet the creature presses on, unsatisfied.",
        "There is a patience here, a stillness that belies the chaos of the screen.",
        "The cursor drifts like a predator scanning the undergrowth for movement.",
        "One can sense the concentration, the narrowing of focus that precedes action.",
        "The rhythm of keystrokes fills the silence like rainfall on dry earth.",
        "The screen reflects in those watchful eyes, endlessly scanning, endlessly seeking.",
        "It is a delicate dance between attention and distraction, played out in pixels.",
        "The creature pauses, hovering, as if sensing something just beyond the fold.",
        "And so the scroll continues, an ancient foraging instinct dressed in modern glass.",
    ]
    if topic == "x":
        detail_templates.extend([
            "The feed scrolls endlessly, a river that never runs dry, and the creature drinks deep.",
            "Post after post flows past, each one a fleeting distraction in an infinite stream.",
        ])
    if topic == "terminal":
        detail_templates.extend([
            "The command line blinks patiently, awaiting the next instruction from its keeper.",
            "In this stark landscape of text and prompts, the creature thrives without color or ornament.",
        ])

    sentence1 = _select_variant(lead_templates, history_lines, turn_index).format(
        app=app, content_cue=content_cue,
    )
    if transition:
        sentence1 = f"{transition}{sentence1[0].lower()}{sentence1[1:]}"
    sentence2 = _select_variant(detail_templates, history_lines, turn_index + 1)

    line = f"{sentence1} {sentence2}"
    return _ascii_only(line)


_STYLE_TEMPLATES: Dict[str, Dict[str, List[str]]] = {
    "horror": {
        "leads": [
            "The cursor drifts toward {app}. Something about {content_cue} feels wrong.",
            "In {app}, {content_cue} loads slowly. Too slowly.",
            "{app} is open. {content_cue} stares back, unblinking.",
            "The screen flickers. {app} is still there. {content_cue} has not changed. Or has it.",
            "They opened {app} again. They always open {app}. {content_cue} is waiting.",
            "A notification appears in {app}. Nobody sent it.",
        ],
        "details": [
            "The tab count has changed. They did not open a new tab.",
            "The scroll position is not where they left it.",
            "There is a window behind {app}. It was not there before.",
            "The loading spinner has been going for too long now.",
            "Something moved in the corner of the screen. Just pixels. Probably.",
            "The cursor hovers over a link. The link hovers back.",
        ],
        "closes": [
            "They do not look away. They should look away.",
            "The screen dims for a moment. When it returns, nothing is different. Nothing at all.",
            "Whatever happens next, the undo button will not save them.",
            "The session continues. It always continues.",
            "They will not remember this moment. But the browser history will.",
            "Somewhere, a process is running that they did not start.",
        ],
    },
    "noir": {
        "leads": [
            "It was another late night in {app}. {content_cue} had the answers, or so I thought.",
            "I found them in {app}, staring at {content_cue} like it owed them money.",
            "{app} again. This case keeps circling back to {content_cue}.",
            "The rain outside matched the mood in {app}. {content_cue} wasnt talking.",
            "I opened {app}. {content_cue} was still there, same as yesterday.",
            "{content_cue} in {app}. The kind of lead that looks good until it doesnt.",
        ],
        "details": [
            "The tabs lined up like suspects in a precinct hallway.",
            "Every click was a door. Every door led to another question.",
            "The search bar blinked, waiting for a confession that wasnt coming.",
            "Scrolling down, looking for something that didnt want to be found.",
            "The browser history read like a rap sheet with no conviction.",
            "The cursor moved slow, deliberate, like it knew more than it was letting on.",
        ],
        "closes": [
            "This case wasnt going anywhere. But neither was I.",
            "The screen glowed. The clock ticked. The truth stayed hidden.",
            "In this town, every answer is just a better question.",
            "I closed the tab. It didnt matter. There would be another one.",
            "Nobody solves a case like this. You just stop asking.",
            "The cursor blinked twice. Even it looked tired.",
        ],
    },
    "reality_tv": {
        "leads": [
            "So {app} is open and honestly? {content_cue} is giving chaos right now.",
            "Ok so like, they are on {app} looking at {content_cue} and I cannot.",
            "Not them back on {app} staring at {content_cue} again. Main character behavior.",
            "{app} is up with {content_cue} on screen and its literally a whole situation.",
            "They just opened {app} and {content_cue} is RIGHT there. The drama.",
            "Here we go. {app}. {content_cue}. This is peak unhinged energy.",
        ],
        "details": [
            "The tabs? A mess. Like their whole digital life is a group project gone wrong.",
            "You can tell they have no plan. Just vibes and a scroll wheel.",
            "This is giving procrastination energy and they do not even care.",
            "The screen is chaos and they are thriving in it. Allegedly.",
            "Not a single bookmark in sight. Living on the edge for real.",
            "They are scrolling like the answer is gonna just appear. Spoiler: it wont.",
        ],
        "closes": [
            "Im not judging. Actually wait, yes I am.",
            "This is the kind of screen time that should come with a warning label.",
            "Production does not endorse this behavior but they definitely filmed it.",
            "Anyway. The mess continues.",
            "If this were a competition they would be losing. Respectfully.",
            "The real villain this season? That tab count.",
        ],
    },
    "asmr": {
        "leads": [
            "And now... we find ourselves in {app}... where {content_cue} glows softly.",
            "Gently now... {app} opens... and there it is... {content_cue}... waiting.",
            "Breathe in... and notice {app}... {content_cue} unfolds before us.",
            "Shhh... {app} is here... and {content_cue} settles into view... so peacefully.",
            "Listen... the soft hum of {app}... as {content_cue} appears on screen.",
            "We drift into {app}... where {content_cue} rests... quiet and patient.",
        ],
        "details": [
            "Each click... so delicate... like a raindrop finding its leaf.",
            "The scroll... smooth and unhurried... pixels flowing like warm honey.",
            "Feel the gentle rhythm of the keys... tap... tap... tap.",
            "The cursor glides... tracing a path only it understands.",
            "A new page loads... and the screen breathes... refreshed.",
            "The soft blue glow of the monitor... wrapping everything in calm.",
        ],
        "closes": [
            "And we are here... and it is enough... and the screen keeps glowing.",
            "Breathe out... the moment passes... another will come.",
            "Everything is exactly where it should be... even the chaos.",
            "The session continues... warm... unhurried... safe.",
            "And so we scroll... gently... endlessly... and that is ok.",
            "Nothing to fix... nothing to rush... just this... just now.",
        ],
    },
    "wrestling": {
        "leads": [
            "BAH GAWD, {app} just entered the ring and {content_cue} is ON FIRE.",
            "OH MY GOD. {app} is OPEN and {content_cue} is taking NO prisoners.",
            "WAIT A MINUTE. Is that {app}? IT IS. And {content_cue} is the main event.",
            "LADIES AND GENTLEMEN, {app} with {content_cue} and this crowd is ELECTRIC.",
            "FROM THE TOP ROPE, its {app} with {content_cue} and NOBODY saw this coming.",
            "THE CHALLENGER {app} enters with {content_cue}. THIS IS NOT A DRILL.",
        ],
        "details": [
            "The tabs are STACKED. Every single one is a contender for the title.",
            "That scroll was DEVASTATING. Three pages in one swipe. UNHEARD OF.",
            "The cursor just did a SUPLEX on that dropdown menu. Beautiful technique.",
            "LOOK at that keystroke speed. This is CHAMPIONSHIP level typing.",
            "A new window SLAMS open. The crowd goes WILD.",
            "That right-click was VICIOUS. The context menu never stood a chance.",
        ],
        "closes": [
            "SOMEBODY STOP THE MATCH. This user has a FAMILY.",
            "AND THE CROWD GOES WILD. What a performance. What a NIGHT.",
            "THAT MAN HAD A FAMILY and he just closed the tab WITHOUT SAVING.",
            "This is HISTORY in the making, folks. HISTORY.",
            "THE REF IS DOWN. THE TABS ARE DOWN. EVERYTHING IS DOWN.",
            "IS IT OVER? NO. The cursor is STILL MOVING. This match continues.",
        ],
    },
}


def _compose_styled_commentary(
    style: str,
    app_name: str,
    window_title: str,
    observations: List[str],
    actions: List[str],
    history_lines: List[str],
    context_history: List[str],
    turn_index: int,
    profanity_level: str,
) -> str:
    """Compose fallback commentary for styles that use template pools."""
    app = app_name or "this app"
    topic = _topic_from_context(app_name, window_title, observations)
    content_cue = _content_cue(app_name, window_title, topic)

    templates = _STYLE_TEMPLATES[style]
    lead = _select_variant(templates["leads"], history_lines, turn_index).format(
        app=app, content_cue=content_cue,
    )
    detail = _select_variant(templates["details"], history_lines, turn_index + 1).format(
        app=app, content_cue=content_cue,
    )
    close = _select_variant(templates["closes"], history_lines, turn_index + 2)

    line = f"{lead} {detail} {close}"
    return _ascii_only(line)


def _compose_commentary(
    app_name: str,
    window_title: str,
    observations: List[str],
    actions: List[str],
    history_lines: List[str],
    context_history: List[str],
    turn_index: int,
    profanity_level: str,
    style: str = "sports",
) -> str:
    if style == "nature":
        return _compose_nature_commentary(
            app_name, window_title, observations, actions,
            history_lines, context_history, turn_index,
        )
    if style in _STYLE_TEMPLATES:
        return _compose_styled_commentary(
            style, app_name, window_title, observations, actions,
            history_lines, context_history, turn_index, profanity_level,
        )
    app = app_name or "this app"
    window = _safe_snippet(window_title)
    topic = _topic_from_context(app_name, window_title, observations)
    allow_ui = not _is_known_app(app_name, window_title, topic)
    obs_pool = _build_observation_pool(observations, topic, allow_ui)
    action_pool = _build_action_pool(actions, topic, profanity_level)

    transition = "In"
    if len(context_history) >= 2:
        prev_app = context_history[-2].split("|", 1)[0].strip()
        if prev_app and prev_app != app:
            transition = "Switching to"
        else:
            transition = "Staying in"

    bridge_options = [
        "Now",
        "Right now",
        "Here we go",
        "Still rolling",
        "Picking it up",
        "Back at it",
        "No break here",
        "Next up",
        "On we go",
        "Still in it",
        "No pause",
        "Staying hot",
        "Clock is ticking",
        "Keeping it moving",
        "Same energy",
        "Riding the wave",
        "No mercy",
        "Look at this",
        "Watch this",
        "Still alive",
    ]
    bridge = _select_variant(bridge_options, history_lines, turn_index)
    action = _select_variant(action_pool, history_lines, turn_index)
    action_clause = _action_clause(action, topic, history_lines, turn_index)

    obs1 = _select_variant(obs_pool, history_lines, turn_index)
    obs2 = _select_variant(obs_pool, history_lines, turn_index + 1) or obs1
    obs3 = _select_variant(obs_pool, history_lines, turn_index + 2) or obs2

    content_cue = _content_cue(app_name, window_title, topic)
    story_lead = ""
    if len(context_history) >= 2:
        prev_app = context_history[-2].split("|", 1)[0].strip()
        if prev_app and prev_app != app:
            lead_templates = [
                "Coming off {prev_app}, ",
                "Fresh off {prev_app}, ",
                "After bouncing out of {prev_app}, ",
                "We just bailed on {prev_app}, ",
                "Leaving {prev_app} behind, ",
                "Stepping away from {prev_app}, ",
                "Walking out of {prev_app}, ",
                "Done with {prev_app} for the moment, ",
                "Shaking off {prev_app}, ",
                "Closing out {prev_app}, ",
                "Moving on from {prev_app}, ",
                "Dropping {prev_app} in the rearview, ",
                "Escaping {prev_app}, ",
                "Cutting away from {prev_app}, ",
            ]
            story_lead = _select_variant(
                [tmpl.format(prev_app=prev_app) for tmpl in lead_templates],
                history_lines,
                turn_index,
            )

    lead_templates = [
        "{bridge}, {transition} {app}{window_part}, {action_clause}.",
        "{bridge}, {transition} {app}{window_part} and {action_clause}.",
        "{bridge}, {transition} {app}{window_part}, {action_clause} like the clock is ticking.",
        "{bridge}, {transition} {app}{window_part} with {action_clause}, no mercy.",
        "{bridge}, {transition} {app}{window_part}, {action_clause} and the tempo stays up.",
        "{bridge}, {transition} {app}{window_part} where its {action_clause} and no rest.",
        "{bridge}, {transition} {app}{window_part} and the move is {action_clause}.",
        "{bridge}, {transition} {app}{window_part}, {action_clause} and the seconds keep slipping.",
        "{bridge}, {transition} {app}{window_part}, {action_clause} with the cursor stalking the play.",
        "{bridge}, {transition} {app}{window_part}, {action_clause} and momentum is still hot.",
    ]
    detail_templates = []
    if allow_ui:
        detail_templates = [
            "You can make out {obs1}, {obs2}, and {obs3} all jockeying for attention.",
            "The screen shows {obs1} and {obs2}, while {obs3} holds the lane.",
            "Visually its {obs1} next to {obs2}, with {obs3} waiting in the wings.",
            "You can spot {obs1}, then {obs2}, and {obs3} hanging off the side.",
            "Front and center is {obs1}, while {obs2} crowds in and {obs3} loiters.",
            "Its {obs1} at the core, {obs2} on the edge, and {obs3} sliding around.",
            "The layout flashes {obs1} with {obs2} nearby, and {obs3} parked in the corner.",
            "Theres {obs1} up top, {obs2} below, and {obs3} just waiting.",
            "You can see {obs1} pressing in, {obs2} holding space, and {obs3} floating around.",
            "Its a mess of {obs1}, {obs2}, and {obs3}, all fighting for the screen.",
            "Theres {obs1} locked in, {obs2} crowding it, and {obs3} hanging off the side.",
            "Its {obs1} stacked against {obs2}, with {obs3} nudging in.",
            "You get {obs1}, then {obs2}, then {obs3}, like a queue of distractions.",
            "Its a clean look: {obs1}, {obs2}, and {obs3} in the mix.",
        ]
    else:
        detail_templates = [
            "Its all about {content_cue}, and the pace is not exactly disciplined.",
            "The focus stays on {content_cue}, while the clock keeps bleeding.",
            "Everything on screen screams {content_cue}, and the momentum keeps rolling.",
            "Its {content_cue} on repeat, and the rest is just scenery.",
            "All eyes on {content_cue} while the minutes drip away.",
            "The whole scene is {content_cue}, no detours, no discipline.",
            "This is {content_cue} and nothing else is driving the bus.",
            "The spotlight is welded to {content_cue}, and time keeps getting cooked.",
            "Its {content_cue} doing laps while the clock waves the flag.",
            "The screen is married to {content_cue} and the pace is reckless.",
            "Its {content_cue} and the rest is just background noise.",
            "All roads lead to {content_cue}, even when they should not.",
        ]
    close_templates = [
        "Feels like a decision is coming, but the scroll wheel keeps winning.",
        "It looks like a move is about to happen, then nope, more browsing.",
        "The cursor hovers, the clock runs, and the next click is still a maybe.",
        "This is the kind of stretch that burns time and pretends its progress.",
        "That next click is loading in slow motion.",
        "Momentum is there, follow-through is missing.",
        "The clock is undefeated and the tab is still open.",
        "This is a detour disguised as direction.",
        "Theres motion, but the finish line is nowhere in sight.",
        "This is the long game of not committing.",
        "The timer keeps running and the play never lands.",
        "It feels productive until you realize it isnt.",
        "You can smell a decision, but the scroll keeps dodging it.",
        "The rhythm stays hot and the resolution stays cold.",
    ]
    if topic == "x":
        close_templates.extend(
            [
                "Over on X, this is straight doomscrolling and a waste of time.",
                "That feed is a damn gravity well, and the scroll wheel knows it.",
                "X is open and the timeline is undefeated right now, wasted minutes and all.",
                "Just burning fucking time in the feed and pretending its research.",
                "The timeline keeps winning and the to-do list keeps losing.",
                "This is the doomscroll Olympics and the gold medal is time wasted.",
                "Every swipe is another minute on the burn pile.",
                "The feed is a vortex and its doing its job.",
            ]
        )
    if topic == "terminal":
        close_templates.extend(
            [
                "Terminal energy all day, no GUI safety net in sight.",
                "This feels like command line cooking, low heat and constant stirring.",
            ]
        )

    sentence1 = _select_variant(lead_templates, history_lines, turn_index).format(
        bridge=bridge,
        transition=transition,
        app=app,
        window_part=_window_part(window),
        action_clause=action_clause,
    )
    if story_lead:
        sentence1 = f"{story_lead}{sentence1[0].lower()}{sentence1[1:]}"
    sentence2 = _select_variant(detail_templates, history_lines, turn_index + 1).format(
        obs1=obs1,
        obs2=obs2,
        obs3=obs3,
        content_cue=content_cue,
    )
    sentence3 = _select_variant(close_templates, history_lines, turn_index + 2)

    line = f"{sentence1} {sentence2} {sentence3}"
    return _ascii_only(line)


_WINDOW_NOISE_PATTERNS = [
    re.compile(r"\s*-\s*High memory usage\s*-\s*\d+\s*MB\s*", re.IGNORECASE),
    re.compile(r"\s*-\s*\d+\s*MB\s*", re.IGNORECASE),
    re.compile(r"\s*-\s*Low memory\s*", re.IGNORECASE),
    re.compile(r"\s*-\s*Not responding\s*", re.IGNORECASE),
]


def _clean_window_title(text: str) -> str:
    """Strip browser chrome noise (memory warnings, etc.) from window titles."""
    if not text:
        return text
    for pattern in _WINDOW_NOISE_PATTERNS:
        text = pattern.sub(" - ", text)
    text = re.sub(r"(\s*-\s*)+", " - ", text).strip(" -")
    return text


def _safe_snippet(text: str, limit: int = 70) -> str:
    if not text:
        return ""
    cleaned = _clean_window_title(text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = re.sub(r"\S+@\S+", "", cleaned)
    cleaned = re.sub(r"\b\d{5,}\b", "", cleaned)
    cleaned = cleaned.replace("\"", "").replace("`", "").replace("\u201c", "").replace("\u201d", "")
    cleaned = _ascii_only(cleaned)
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 3].rsplit(" ", 1)[0] + "..."
    return cleaned


def _ensure_context(
    line: Optional[str], app_name: str, window_title: str, style: str = "",
) -> Optional[str]:
    """Pass through — the prompt tells Gemini to reference context naturally."""
    return line


def _ensure_punctuation(line: Optional[str]) -> Optional[str]:
    if not line:
        return line
    if _ends_with_punctuation(line):
        return line
    return f"{line}."


def _expand_line(
    line: Optional[str],
    config: AppConfig,
    app_name: str,
    window_title: str,
    observations: List[str],
    actions: List[str],
    history_lines: List[str],
    context_history: List[str],
    turn_index: int,
) -> Optional[str]:
    if not line:
        return line
    extras: List[str] = []
    topic = _topic_from_context(app_name, window_title, observations)
    allow_ui = not _is_known_app(app_name, window_title, topic)
    obs_pool = _build_observation_pool(observations, topic, allow_ui)
    action_pool = _build_action_pool(actions, topic, config.profanity_level)

    if obs_pool:
        obs1 = _select_variant(obs_pool, history_lines, turn_index)
        obs2 = _select_variant(obs_pool, history_lines, turn_index + 1) or obs1
        extra = f"On screen, {obs1} and {obs2} pull focus."
        if extra.lower() not in line.lower():
            extras.append(extra)
    elif not allow_ui:
        cue = _content_cue(app_name, window_title, topic)
        if cue.lower() not in line.lower():
            extras.append(
                f"The moment stays on {cue}, and the seconds keep slipping."
            )

    if action_pool:
        action = _select_variant(action_pool, history_lines, turn_index)
        extras.append(f"The main move right now is {action}, and it stays deliberate.")

    window = _safe_snippet(window_title)
    if window and window.lower() not in line.lower():
        extras.append(f"The window title reads {window}, keeping the story grounded.")

    if context_history:
        prev_app = (
            context_history[-2].split("|", 1)[0].strip() if len(context_history) > 1 else ""
        )
        if prev_app and prev_app != app_name:
            extras.append(f"That switch into {app_name} shifts the rhythm, but the focus holds.")

    if _too_similar(line, history_lines):
        flair = _select_variant(_flair_pool(topic, config.profanity_level, config.narration_style), history_lines, turn_index)
        if flair:
            extras.insert(0, flair)

    expanded = line
    for extra in extras:
        if _meets_constraints(expanded, config, app_name, window_title):
            return expanded
        expanded = f"{expanded} {extra}"

    if _meets_constraints(expanded, config, app_name, window_title):
        return expanded

    if not _ends_with_punctuation(expanded):
        expanded = f"{expanded}."
    return expanded


def _lowercase_initial(text: str) -> str:
    if not text:
        return text
    return text[0].lower() + text[1:]


def _ascii_only(text: str) -> str:
    return text.encode("ascii", "ignore").decode("ascii")
