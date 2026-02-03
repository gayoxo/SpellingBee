import os
import random
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

import speech_recognition as sr
from dotenv import load_dotenv
import pyttsx3


# ==========================
# .env / Config helpers
# ==========================
load_dotenv()

def env_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key, str(default)).strip().lower()
    return v in ("1", "true", "yes", "y", "on")

def env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)).strip())
    except Exception:
        return default

def env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)).strip())
    except Exception:
        return default


# ==========================
# Config from .env
# ==========================
WORDS_FILE = os.getenv("WORDS_FILE", "words.txt")
ROUNDS_DEFAULT = env_int("ROUNDS_DEFAULT", 10)

EXAM_MODE = env_bool("EXAM_MODE", True)
SPEAK_WORD_ON_NEW_ROUND = env_bool("SPEAK_WORD_ON_NEW_ROUND", True)

AUTO_LISTEN_WINDOW = env_int("AUTO_LISTEN_WINDOW", 45)
AUTO_LISTEN_CHUNK = env_int("AUTO_LISTEN_CHUNK", 6)

MIN_LETTERS_VALID = env_int("MIN_LETTERS_VALID", 2)
MIN_RELATIVE_VALID = env_float("MIN_RELATIVE_VALID", 0.35)  # 0 disables
MAX_RELATIVE_REQUIRED = env_int("MAX_RELATIVE_REQUIRED", 8)

AMBIENT_DURATION = env_float("AMBIENT_DURATION", 1.0)

SCORING_MODE = os.getenv("SCORING_MODE", "edit").strip().lower()  # position | edit
PENALIZE_EXTRA = env_bool("PENALIZE_EXTRA", True)
NO_ATTEMPT_COUNTS_AS_ROUND = env_bool("NO_ATTEMPT_COUNTS_AS_ROUND", True)

WINDOW_WIDTH = env_int("WINDOW_WIDTH", 760)
WINDOW_HEIGHT = env_int("WINDOW_HEIGHT", 520)
SHOW_DETAILS = env_bool("SHOW_DETAILS", True)

TTS_RATE = env_int("TTS_RATE", 130)
TTS_VOLUME = env_float("TTS_VOLUME", 1.0)
TTS_VOICE_HINT = os.getenv("TTS_VOICE_HINT", "en").strip().lower()


# ==========================
# Letters normalization
# ==========================
LETTER_MAP = {
    "a": "a", "ay": "a",
    "b": "b", "bee": "b",
    "c": "c", "see": "c", "sea": "c",
    "d": "d", "dee": "d",
    "e": "e", "ee": "e",
    "f": "f", "eff": "f",
    "g": "g", "gee": "g",
    "h": "h", "aitch": "h",
    "i": "i", "eye": "i",
    "j": "j", "jay": "j",
    "k": "k", "kay": "k",
    "l": "l", "el": "l",
    "m": "m", "em": "m",
    "n": "n", "en": "n",
    "o": "o", "oh": "o",
    "p": "p", "pee": "p",
    "q": "q", "cue": "q",
    "r": "r", "are": "r",
    "s": "s", "ess": "s",
    "t": "t", "tee": "t",
    "u": "u", "you": "u",
    "v": "v", "vee": "v",
    "w": "w", "double u": "w",
    "x": "x", "ex": "x",
    "y": "y", "why": "y",
    "z": "z", "zee": "z", "zed": "z",
}
NOISE_WORDS = {"letter", "capital", "small", "uppercase", "lowercase"}


def normalize_letters(text: str) -> str:
    """
    Convert ASR transcript to a string of letters.
    Examples:
      "a pee pee el ee" -> "apple"
      "double u eye" -> "wi"
    """
    text = (text or "").lower().replace("-", " ")
    tokens = text.split()
    result = []

    i = 0
    while i < len(tokens):
        bigram = f"{tokens[i]} {tokens[i+1]}" if i + 1 < len(tokens) else None
        if bigram and bigram in LETTER_MAP:
            result.append(LETTER_MAP[bigram])
            i += 2
            continue

        tok = tokens[i]
        if tok in NOISE_WORDS:
            i += 1
            continue

        if tok in LETTER_MAP:
            result.append(LETTER_MAP[tok])
        elif len(tok) == 1 and tok.isalpha():
            result.append(tok)

        i += 1

    return "".join(result)


# ==========================
# Scoring
# ==========================
def score_positionwise(target: str, guess: str) -> dict:
    """
    Simple per-index scoring. Can fail when there's a missing letter that shifts the rest.
    """
    t = (target or "").lower().strip()
    g = (guess or "").lower().strip()

    max_len = max(len(t), len(g))
    correct = 0
    per_pos = []

    for idx in range(max_len):
        tc = t[idx] if idx < len(t) else None
        gc = g[idx] if idx < len(g) else None
        ok = (tc == gc) and (tc is not None)
        correct += 1 if ok else 0
        per_pos.append((idx, tc, gc, ok))

    denom = max_len if PENALIZE_EXTRA else max(len(t), 1)
    accuracy = (correct / denom) * 100 if denom else 100.0
    missing_suffix = t[len(g):] if len(g) < len(t) else ""

    return {
        "target": t,
        "guess": g,
        "accuracy": accuracy,
        "correct_positions": correct,
        "denom": denom,
        "missing_suffix": missing_suffix,
        "per_pos": per_pos,
        "exact": (t == g),
    }


def score_edit_distance(target: str, guess: str) -> dict:
    """
    Dynamic alignment using Levenshtein distance with backtrace.
    Operations:
      '=' match, 'S' substitution, 'I' insertion (extra in guess), 'D' deletion (missing from guess)
    Accuracy:
      matches / len(target)
    """
    t = (target or "").lower().strip()
    g = (guess or "").lower().strip()

    n, m = len(t), len(g)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    back = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        back[i][0] = ("D", i - 1, 0)
    for j in range(1, m + 1):
        dp[0][j] = j
        back[0][j] = ("I", 0, j - 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost_sub = 0 if t[i - 1] == g[j - 1] else 1

            cand_sub = dp[i - 1][j - 1] + cost_sub
            cand_del = dp[i - 1][j] + 1
            cand_ins = dp[i][j - 1] + 1

            best = cand_sub
            op = "=" if cost_sub == 0 else "S"
            prev = (i - 1, j - 1)

            if cand_del < best:
                best = cand_del
                op = "D"
                prev = (i - 1, j)

            if cand_ins < best:
                best = cand_ins
                op = "I"
                prev = (i, j - 1)

            dp[i][j] = best
            back[i][j] = (op, prev[0], prev[1])

    # backtrace
    aligned = []  # (expected_char|None, got_char|None, op)
    i, j = n, m
    while i > 0 or j > 0:
        op, pi, pj = back[i][j]
        if op in ("=", "S"):
            aligned.append((t[i - 1], g[j - 1], op))
        elif op == "D":
            aligned.append((t[i - 1], None, op))
        elif op == "I":
            aligned.append((None, g[j - 1], op))
        i, j = pi, pj
    aligned.reverse()

    matches = sum(1 for tc, gc, op in aligned if op == "=")
    substitutions = sum(1 for tc, gc, op in aligned if op == "S")
    deletions = sum(1 for tc, gc, op in aligned if op == "D")
    insertions = sum(1 for tc, gc, op in aligned if op == "I")
    dist = dp[n][m]

    denom = max(len(t), 1)
    accuracy = (matches / denom) * 100.0

    missing_letters = "".join(tc for tc, gc, op in aligned if op == "D" and tc)

    return {
        "target": t,
        "guess": g,
        "accuracy": accuracy,
        "matches": matches,
        "denom": denom,
        "edit_distance": dist,
        "insertions": insertions,
        "deletions": deletions,
        "substitutions": substitutions,
        "missing_letters": missing_letters,
        "aligned": aligned,
        "exact": (t == g),
    }


def format_alignment_details_edit(result: dict) -> str:
    lines = []
    lines.append(f"Target : {result['target']}")
    lines.append(f"Guess  : {result['guess']}")
    lines.append(
        f"Edits  : dist={result['edit_distance']} "
        f"(D={result['deletions']}, I={result['insertions']}, S={result['substitutions']})"
    )
    if result.get("missing_letters"):
        lines.append(f"Missing letters: {result['missing_letters']}")
    lines.append("")
    lines.append("Idx | expected | got | op")
    lines.append("-" * 26)

    for idx, (tc, gc, op) in enumerate(result["aligned"]):
        tc_disp = tc if tc is not None else "∅"
        gc_disp = gc if gc is not None else "∅"
        lines.append(f"{idx:02d}  |   {tc_disp}      |  {gc_disp}  | {op}")

    return "\n".join(lines)


def format_alignment_details_position(result: dict) -> str:
    lines = []
    lines.append(f"Target : {result['target']}")
    lines.append(f"Guess  : {result['guess']}")
    lines.append("")
    lines.append("Pos | expected | got | ok")
    lines.append("-" * 26)
    for i, tc, gc, ok in result["per_pos"]:
        tc_disp = tc if tc is not None else "∅"
        gc_disp = gc if gc is not None else "∅"
        mark = "✓" if ok else "x"
        lines.append(f"{i:02d}  |   {tc_disp}      |  {gc_disp}  | {mark}")
    return "\n".join(lines)


# ==========================
# App
# ==========================
class SpellingBeeApp(tk.Tk):
    def __init__(self, words_file: str):
        super().__init__()

        self.title("Spelling Bee (Start + Auto Listen + Next + Edit Alignment)")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.minsize(680, 440)

        self.words = self._load_words(words_file)

        self.rounds_total = tk.IntVar(value=ROUNDS_DEFAULT)
        self.rounds_done = 0
        self.score_sum = 0.0
        self.current_word = ""

        self.recognizer = sr.Recognizer()

        self.tts = pyttsx3.init()
        self.tts.setProperty("rate", TTS_RATE)
        self.tts.setProperty("volume", TTS_VOLUME)
        self._try_set_english_voice()

        self._stop_listening = threading.Event()

        self.app_started = False
        self.round_state = "idle"  # idle | listening | finished

        self._build_ui()
        self._set_idle_screen()

    # ---------- data ----------
    def _load_words(self, path: str) -> list[str]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"No existe el fichero de palabras: {p.resolve()}")

        words = []
        for line in p.read_text(encoding="utf-8").splitlines():
            w = line.strip().lower()
            if w and not w.startswith("#"):
                words.append(w)

        if not words:
            raise ValueError("El fichero de palabras está vacío (o solo contiene comentarios/líneas vacías).")

        return words

    # ---------- TTS ----------
    def _try_set_english_voice(self):
        try:
            voices = self.tts.getProperty("voices") or []
            for v in voices:
                blob = f"{getattr(v, 'id', '')} {getattr(v, 'name', '')}".lower()
                if TTS_VOICE_HINT and TTS_VOICE_HINT in blob:
                    self.tts.setProperty("voice", v.id)
                    break
        except Exception:
            pass

    def speak_word(self, word: str):
        try:
            self.tts.say(word)
            self.tts.runAndWait()
        except Exception:
            pass

    # ---------- UI ----------
    def _build_ui(self):
        pad = 10

        top = ttk.Frame(self, padding=pad)
        top.pack(fill="x")
        ttk.Label(
            top,
            text="Spelling Bee — Start → pronuncia → auto escucha → Siguiente (scoring por alineamiento)",
            font=("Segoe UI", 14, "bold")
        ).pack(anchor="w")

        conf = ttk.Frame(self, padding=pad)
        conf.pack(fill="x")

        ttk.Label(conf, text="Rondas:").grid(row=0, column=0, sticky="w")
        self.rounds_spin = ttk.Spinbox(conf, from_=1, to=50, textvariable=self.rounds_total, width=6)
        self.rounds_spin.grid(row=0, column=1, sticky="w", padx=(6, 18))

        self.btn_start = ttk.Button(conf, text="▶ Start", command=self.on_start)
        self.btn_start.grid(row=0, column=2, padx=6)

        self.btn_speak = ttk.Button(conf, text="🔊 Repetir palabra", command=lambda: self.speak_word(self.current_word))
        self.btn_speak.grid(row=0, column=3, padx=6)

        self.btn_reset = ttk.Button(conf, text="Reset", command=self.reset_game)
        self.btn_reset.grid(row=0, column=4, padx=6)

        word_frame = ttk.LabelFrame(self, text="Palabra / Audio", padding=pad)
        word_frame.pack(fill="x", padx=pad, pady=(0, pad))

        self.word_var = tk.StringVar(value="")
        ttk.Label(word_frame, textvariable=self.word_var, font=("Consolas", 22, "bold")).pack(anchor="center")

        ttk.Label(
            word_frame,
            text=f"Ventana de escucha: {AUTO_LISTEN_WINDOW}s (chunks {AUTO_LISTEN_CHUNK}s). "
                 f"Válido: ≥{MIN_LETTERS_VALID} letras y ≥{MIN_RELATIVE_VALID:.2f}·len(word) (máx {MAX_RELATIVE_REQUIRED})."
        ).pack(anchor="w", pady=(8, 0))

        res = ttk.LabelFrame(self, text="Resultado", padding=pad)
        res.pack(fill="both", expand=True, padx=pad, pady=(0, pad))

        self.status_var = tk.StringVar(value="")
        ttk.Label(res, textvariable=self.status_var, font=("Segoe UI", 11, "bold")).pack(anchor="w")

        self.raw_var = tk.StringVar(value="Heard (raw): -")
        self.norm_var = tk.StringVar(value="Heard (letters): -")
        ttk.Label(res, textvariable=self.raw_var).pack(anchor="w", pady=(6, 0))
        ttk.Label(res, textvariable=self.norm_var).pack(anchor="w")

        self.summary_var = tk.StringVar(value="Accuracy: -")
        ttk.Label(res, textvariable=self.summary_var, font=("Segoe UI", 11)).pack(anchor="w", pady=(8, 0))

        self.details = tk.Text(res, height=12, wrap="none")
        self.details.pack(fill="both", expand=True, pady=(10, 0))
        self.details.configure(state="disabled")

        footer = ttk.Frame(self, padding=pad)
        footer.pack(fill="x")

        self.progress_var = tk.StringVar(value="Ronda: 0/0 | Media: 0.0%")
        ttk.Label(footer, textvariable=self.progress_var).pack(side="left")

        self.btn_next = ttk.Button(footer, text="➡️ Siguiente", command=self.on_next)
        self.btn_next.pack(side="right", padx=(6, 0))

        self.btn_finish = ttk.Button(footer, text="Finalizar", command=self.finish_game)
        self.btn_finish.pack(side="right")

        self._set_next_enabled(False)

    def _set_details(self, text: str):
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", text)
        self.details.configure(state="disabled")

    def _update_progress(self):
        total = int(self.rounds_total.get())
        avg = (self.score_sum / self.rounds_done) if self.rounds_done else 0.0
        self.progress_var.set(f"Ronda: {self.rounds_done}/{total} | Media: {avg:.1f}%")

    def _update_word_display(self):
        if not self.app_started:
            self.word_var.set("Pulsa ▶ Start para comenzar")
            return
        if not EXAM_MODE:
            self.word_var.set(self.current_word)
        else:
            self.word_var.set("🔊 Listen and spell (auto listening)")

    def _set_next_enabled(self, enabled: bool):
        self.btn_next.config(state=("normal" if enabled else "disabled"))

    def _disable_controls_while_listening(self, listening: bool):
        state_other = "disabled" if listening else "normal"
        self.rounds_spin.config(state=state_other)
        self.btn_reset.config(state=state_other)
        self.btn_finish.config(state=state_other)
        self.btn_speak.config(state=state_other)
        self.btn_start.config(state=("disabled" if self.app_started else "normal"))
        if listening:
            self._set_next_enabled(False)

    def _set_idle_screen(self):
        self.app_started = False
        self.round_state = "idle"
        self._stop_listening.set()

        self.current_word = ""
        self._update_word_display()
        self.status_var.set("Listo. Pulsa ▶ Start.")
        self.raw_var.set("Heard (raw): -")
        self.norm_var.set("Heard (letters): -")
        self.summary_var.set("Accuracy: -")
        self._set_details("")
        self._update_progress()
        self._set_next_enabled(False)

        self.btn_start.config(state="normal")
        self.btn_speak.config(state="disabled")

    # ---------- Start / Next ----------
    def on_start(self):
        if self.app_started:
            return
        self.app_started = True
        self.btn_start.config(state="disabled")
        self.btn_speak.config(state="normal")
        self.start_new_round()

    def on_next(self):
        if not self.app_started or self.round_state != "finished":
            return

        total = int(self.rounds_total.get())
        if self.rounds_done >= total:
            self.finish_game(auto=True)
            return

        self.start_new_round()

    # ---------- Speech to text ----------
    def _listen_google_en_chunk(self) -> str:
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=AMBIENT_DURATION)
            audio = self.recognizer.listen(source, phrase_time_limit=AUTO_LISTEN_CHUNK)
        return self.recognizer.recognize_google(audio, language="en-US").lower()

    def _is_valid_attempt(self, letters: str) -> bool:
        if not letters:
            return False

        if len(letters) < max(1, MIN_LETTERS_VALID):
            return False

        if MIN_RELATIVE_VALID and MIN_RELATIVE_VALID > 0.0:
            needed = max(1, int(len(self.current_word) * MIN_RELATIVE_VALID))
            needed = min(needed, max(1, MAX_RELATIVE_REQUIRED))
            return len(letters) >= needed

        return True

    def _start_auto_listen_window(self):
        self._stop_listening.clear()
        deadline = time.time() + max(1, AUTO_LISTEN_WINDOW)

        last_raw = None
        last_letters = None

        while time.time() < deadline and not self._stop_listening.is_set():
            try:
                raw = self._listen_google_en_chunk()
                letters = normalize_letters(raw)
                last_raw, last_letters = raw, letters

                if self._is_valid_attempt(letters):
                    self.after(0, lambda r=raw, l=letters: self._handle_attempt(r, l))
                    return
                else:
                    self.after(0, lambda r=raw, l=letters: self._update_heard_preview(r, l))

            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                self.after(0, lambda: self._handle_stt_error(f"Error del servicio de reconocimiento:\n{e}"))
                return
            except Exception as e:
                self.after(0, lambda: self._handle_stt_error(f"Error inesperado:\n{e}"))
                return

        self.after(0, lambda r=last_raw, l=last_letters: self._handle_no_attempt(r, l))

    def _update_heard_preview(self, raw: str, letters: str):
        self.raw_var.set(f"Heard (raw): {raw}")
        self.norm_var.set(f"Heard (letters): {letters if letters else '(vacío)'}")
        self.summary_var.set("Accuracy: - (listening...)")

    def _handle_stt_error(self, msg: str):
        self.round_state = "finished"
        self._disable_controls_while_listening(False)
        messagebox.showerror("STT Error", msg)
        self.status_var.set("🌐 Error STT. Pulsa ➡️ Siguiente para continuar.")
        self._set_next_enabled(True)

    # ---------- Round lifecycle ----------
    def reset_game(self):
        self._stop_listening.set()
        self.rounds_done = 0
        self.score_sum = 0.0
        self._set_idle_screen()

    def start_new_round(self):
        self._stop_listening.set()

        self.current_word = random.choice(self.words)
        self._update_word_display()
        self._update_progress()

        self.raw_var.set("Heard (raw): -")
        self.norm_var.set("Heard (letters): -")
        self.summary_var.set("Accuracy: -")
        self._set_details("")
        self._set_next_enabled(False)

        self.round_state = "listening"
        self._disable_controls_while_listening(True)

        self.status_var.set("🔊 Pronunciando palabra…")
        self.update_idletasks()

        if SPEAK_WORD_ON_NEW_ROUND:
            self.speak_word(self.current_word)

        self.status_var.set(f"🎙️ Escuchando hasta {AUTO_LISTEN_WINDOW}s… (deletrea ahora)")
        t = threading.Thread(target=self._start_auto_listen_window, daemon=True)
        t.start()

    def _handle_no_attempt(self, raw_last: str | None, letters_last: str | None):
        self.round_state = "finished"
        self._disable_controls_while_listening(False)

        self.raw_var.set(f"Heard (raw): {raw_last}" if raw_last else "Heard (raw): -")
        if letters_last:
            self.norm_var.set(f"Heard (letters): {letters_last} (not valid)")
        else:
            self.norm_var.set("Heard (letters): (no valid spelling)")

        self.summary_var.set("Accuracy: 0.0% (no attempt)")
        self.status_var.set("⏱️ Tiempo agotado: no se detectó deletreo válido. Pulsa ➡️ Siguiente.")

        if NO_ATTEMPT_COUNTS_AS_ROUND:
            self.rounds_done += 1
            self.score_sum += 0.0
            self._update_progress()

        self._set_next_enabled(True)

    def _score(self, target: str, guess: str) -> dict:
        if SCORING_MODE == "edit":
            return score_edit_distance(target, guess)
        return score_positionwise(target, guess)

    def _handle_attempt(self, raw_text: str, letters: str):
        self.round_state = "finished"
        self._disable_controls_while_listening(False)

        self.raw_var.set(f"Heard (raw): {raw_text}")
        self.norm_var.set(f"Heard (letters): {letters}")

        result = self._score(self.current_word, letters)

        if SCORING_MODE == "edit":
            self.summary_var.set(
                f"Accuracy: {result['accuracy']:.1f}%  "
                f"(matches {result['matches']}/{result['denom']}, dist={result['edit_distance']})"
            )
        else:
            self.summary_var.set(
                f"Accuracy: {result['accuracy']:.1f}%  "
                f"({result['correct_positions']}/{result['denom']} posiciones)"
            )

        if result.get("exact"):
            self.status_var.set("✅ Perfect! Pulsa ➡️ Siguiente.")
        else:
            self.status_var.set("⚠️ Not perfect. Pulsa ➡️ Siguiente.")

        if SHOW_DETAILS:
            if SCORING_MODE == "edit":
                self._set_details(format_alignment_details_edit(result))
            else:
                self._set_details(format_alignment_details_position(result))
        else:
            self._set_details("")

        self.rounds_done += 1
        self.score_sum += float(result["accuracy"])
        self._update_progress()

        self._set_next_enabled(True)

    def finish_game(self, auto: bool = False):
        self._stop_listening.set()

        total = int(self.rounds_total.get())
        if self.rounds_done == 0:
            messagebox.showinfo("Resultado", "Aún no has completado ninguna ronda.")
            return

        avg = self.score_sum / self.rounds_done
        title = "Rondas completadas" if auto else "Finalizado"
        messagebox.showinfo(title, f"Rondas: {self.rounds_done}/{total}\nMedia final: {avg:.1f}%")


if __name__ == "__main__":
    try:
        app = SpellingBeeApp(words_file=WORDS_FILE)
        app.mainloop()
    except Exception as e:
        try:
            messagebox.showerror("Error al iniciar", str(e))
        except Exception:
            print("Error al iniciar:", e)
