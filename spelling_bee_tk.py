import os
import random
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

# Exam mode: do not show the word; speak it in English
EXAM_MODE = env_bool("EXAM_MODE", True)
SPEAK_WORD_ON_NEW_ROUND = env_bool("SPEAK_WORD_ON_NEW_ROUND", True)

# Audio recording parameters (speech to text)
LISTEN_TIME_LIMIT = env_int("LISTEN_TIME_LIMIT", 6)
AMBIENT_DURATION = env_float("AMBIENT_DURATION", 0.6)

# UI
WINDOW_WIDTH = env_int("WINDOW_WIDTH", 760)
WINDOW_HEIGHT = env_int("WINDOW_HEIGHT", 520)
SHOW_DETAILS = env_bool("SHOW_DETAILS", True)

# Scoring
SCORING_MODE = os.getenv("SCORING_MODE", "position").strip().lower()   # position | (future)
PENALIZE_EXTRA = env_bool("PENALIZE_EXTRA", True)

# TTS (Text-to-Speech)
TTS_RATE = env_int("TTS_RATE", 170)
TTS_VOLUME = env_float("TTS_VOLUME", 1.0)
TTS_VOICE_HINT = os.getenv("TTS_VOICE_HINT", "en").strip().lower()  # try "en"


# ==========================
# Spelling: letters mode (normalization)
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
    Convert ASR transcript to letters string. Examples:
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


def score_positionwise(target: str, guess: str) -> dict:
    """
    Position-wise score:
      correct_positions = count matches by index
      accuracy = correct_positions / denom
    denom:
      - if PENALIZE_EXTRA: denom = max(len(target), len(guess))
      - else denom = len(target)  (extra letters do not reduce denom)
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


def load_words(path: str) -> list[str]:
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


# ==========================
# Tkinter App
# ==========================
class SpellingBeeApp(tk.Tk):
    def __init__(self, words_file: str):
        super().__init__()

        self.title("Spelling Bee (Voice Letters)")
        self.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.minsize(680, 440)

        self.words = load_words(words_file)

        # Game state
        self.rounds_total = tk.IntVar(value=ROUNDS_DEFAULT)
        self.rounds_done = 0
        self.score_sum = 0.0
        self.current_word = ""

        # Speech-to-text
        self.recognizer = sr.Recognizer()

        # TTS (offline)
        self.tts = pyttsx3.init()
        self.tts.setProperty("rate", TTS_RATE)
        self.tts.setProperty("volume", TTS_VOLUME)
        self._try_set_english_voice()

        self._build_ui()
        self.new_round()

    # ---------- TTS ----------
    def _try_set_english_voice(self):
        """
        Best-effort: pick a voice that includes 'en' (or your hint) in its id/name.
        If none found, keep default system voice.
        """
        try:
            voices = self.tts.getProperty("voices") or []
            chosen = None
            for v in voices:
                blob = f"{getattr(v, 'id', '')} {getattr(v, 'name', '')}".lower()
                if TTS_VOICE_HINT and TTS_VOICE_HINT in blob:
                    chosen = v.id
                    break
            if chosen:
                self.tts.setProperty("voice", chosen)
        except Exception:
            pass

    def speak_word(self, word: str):
        """
        Speak a word using offline TTS. Non-blocking would be nicer,
        but for simplicity we keep it blocking (short utterances).
        """
        try:
            self.tts.say(word)
            self.tts.runAndWait()
        except Exception:
            # Do not break the game if TTS fails
            pass

    # ---------- UI ----------
    def _build_ui(self):
        pad = 10

        top = ttk.Frame(self, padding=pad)
        top.pack(fill="x")
        ttk.Label(
            top,
            text="Spelling Bee — modo letras por micrófono",
            font=("Segoe UI", 14, "bold")
        ).pack(anchor="w")

        conf = ttk.Frame(self, padding=pad)
        conf.pack(fill="x")

        ttk.Label(conf, text="Rondas:").grid(row=0, column=0, sticky="w")
        self.rounds_spin = ttk.Spinbox(conf, from_=1, to=50, textvariable=self.rounds_total, width=6)
        self.rounds_spin.grid(row=0, column=1, sticky="w", padx=(6, 18))

        self.btn_new = ttk.Button(conf, text="Nueva palabra", command=self.new_round)
        self.btn_new.grid(row=0, column=2, padx=6)

        self.btn_listen = ttk.Button(conf, text="🎙️ Escuchar", command=self.listen_and_check)
        self.btn_listen.grid(row=0, column=3, padx=6)

        self.btn_repeat = ttk.Button(conf, text="Repetir (no cuenta)", command=self.repeat_listen_no_score)
        self.btn_repeat.grid(row=0, column=4, padx=6)

        self.btn_speak = ttk.Button(conf, text="🔊 Repetir palabra", command=lambda: self.speak_word(self.current_word))
        self.btn_speak.grid(row=0, column=5, padx=6)

        self.btn_reset = ttk.Button(conf, text="Reset", command=self.reset_game)
        self.btn_reset.grid(row=0, column=6, padx=6)

        word_frame = ttk.LabelFrame(self, text="Palabra / Audio", padding=pad)
        word_frame.pack(fill="x", padx=pad, pady=(0, pad))

        self.word_var = tk.StringVar(value="")
        self.word_label = ttk.Label(word_frame, textvariable=self.word_var, font=("Consolas", 22, "bold"))
        self.word_label.pack(anchor="center")

        ttk.Label(
            word_frame,
            text="Deletrea diciendo letras en inglés separadas: “A P P L E”, “double u”, “zee/zed”…"
        ).pack(anchor="w", pady=(8, 0))

        res = ttk.LabelFrame(self, text="Resultado", padding=pad)
        res.pack(fill="both", expand=True, padx=pad, pady=(0, pad))

        self.status_var = tk.StringVar(value="Pulsa 🎙️ Escuchar para deletrear.")
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

        self.btn_finish = ttk.Button(footer, text="Finalizar", command=self.finish_game)
        self.btn_finish.pack(side="right")

    # ---------- Game logic ----------
    def reset_game(self):
        self.rounds_done = 0
        self.score_sum = 0.0
        self.raw_var.set("Heard (raw): -")
        self.norm_var.set("Heard (letters): -")
        self.summary_var.set("Accuracy: -")
        self.status_var.set("Juego reiniciado.")
        self._set_details("")
        self.new_round()

    def new_round(self):
        self.current_word = random.choice(self.words)
        self._update_word_display()

        # In exam mode, the hint is the spoken word
        if EXAM_MODE and SPEAK_WORD_ON_NEW_ROUND:
            self.speak_word(self.current_word)

        self.status_var.set("Pulsa 🎙️ Escuchar y deletrea la palabra letra a letra.")
        self._update_progress()

    def _update_word_display(self):
        if not EXAM_MODE:
            self.word_var.set(self.current_word)
        else:
            # Exam mode: do not show the word
            self.word_var.set("🔊 Listen and spell (voice letters)")

    def _update_progress(self):
        total = int(self.rounds_total.get())
        avg = (self.score_sum / self.rounds_done) if self.rounds_done else 0.0
        self.progress_var.set(f"Ronda: {self.rounds_done}/{total} | Media: {avg:.1f}%")

    def _set_details(self, text: str):
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", text)
        self.details.configure(state="disabled")

    # ---------- Audio (STT) ----------
    def _listen_google_en(self) -> str:
        """
        Listen and transcribe with Google (needs internet).
        Controlled by .env: AMBIENT_DURATION, LISTEN_TIME_LIMIT
        """
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=AMBIENT_DURATION)
            audio = self.recognizer.listen(source, phrase_time_limit=LISTEN_TIME_LIMIT)

        return self.recognizer.recognize_google(audio, language="en-US").lower()

    def _disable_buttons(self, disabled: bool):
        state = "disabled" if disabled else "normal"
        self.btn_listen.config(state=state)
        self.btn_repeat.config(state=state)
        self.btn_new.config(state=state)
        self.btn_reset.config(state=state)
        self.btn_speak.config(state=state)

    def repeat_listen_no_score(self):
        """
        Listen and show transcript, but do not count a round nor score.
        """
        self._disable_buttons(True)
        self.status_var.set("🎙️ Escuchando (repetir, no cuenta)...")
        self.update_idletasks()

        try:
            raw_text = self._listen_google_en()
        except sr.UnknownValueError:
            self.status_var.set("🤷 No pude entender el audio. Repite más despacio.")
            self._disable_buttons(False)
            return
        except sr.RequestError as e:
            messagebox.showerror("Error reconocimiento", f"Error del servicio de reconocimiento:\n{e}")
            self.status_var.set("🌐 Error del servicio de reconocimiento.")
            self._disable_buttons(False)
            return
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error:\n{e}")
            self.status_var.set("❌ Error inesperado.")
            self._disable_buttons(False)
            return

        normalized = normalize_letters(raw_text)
        self.raw_var.set(f"Heard (raw): {raw_text}")
        self.norm_var.set(f"Heard (letters): {normalized if normalized else '(vacío)'}")
        self.summary_var.set("Accuracy: - (no cuenta)")
        self.status_var.set("🔁 Repetición mostrada (no cuenta). Pulsa 🎙️ Escuchar para evaluar.")

        if SHOW_DETAILS:
            self._set_details("Repetición sin evaluación.\nPulsa 🎙️ Escuchar para evaluar este intento.")
        else:
            self._set_details("")

        self._disable_buttons(False)

    def listen_and_check(self):
        if not self.current_word:
            self.new_round()

        self._disable_buttons(True)
        self.status_var.set("🎙️ Escuchando... (habla ahora)")
        self.update_idletasks()

        try:
            raw_text = self._listen_google_en()
        except sr.UnknownValueError:
            self.status_var.set("🤷 No pude entender el audio. Prueba otra vez.")
            self._disable_buttons(False)
            return
        except sr.RequestError as e:
            messagebox.showerror("Error reconocimiento", f"Error del servicio de reconocimiento:\n{e}")
            self.status_var.set("🌐 Error del servicio de reconocimiento.")
            self._disable_buttons(False)
            return
        except Exception as e:
            messagebox.showerror("Error", f"Ocurrió un error:\n{e}")
            self.status_var.set("❌ Error inesperado.")
            self._disable_buttons(False)
            return

        normalized = normalize_letters(raw_text)

        self.raw_var.set(f"Heard (raw): {raw_text}")
        self.norm_var.set(f"Heard (letters): {normalized if normalized else '(vacío)'}")

        # Scoring
        if SCORING_MODE != "position":
            result = score_positionwise(self.current_word, normalized)
        else:
            result = score_positionwise(self.current_word, normalized)

        self.summary_var.set(
            f"Accuracy: {result['accuracy']:.1f}%  "
            f"({result['correct_positions']}/{result['denom']} posiciones)"
        )

        if result["exact"]:
            self.status_var.set("✅ Perfect!")
        else:
            msg = "⚠️ Not perfect."
            if result["missing_suffix"]:
                msg += f" Missing at end: '{result['missing_suffix']}'"
            self.status_var.set(msg)

        if SHOW_DETAILS:
            details_lines = []
            details_lines.append(f"Target : {result['target']}")
            details_lines.append(f"Guess  : {result['guess']}")
            details_lines.append("")
            details_lines.append("Pos | expected | got | ok")
            details_lines.append("-" * 26)
            for i, tc, gc, ok in result["per_pos"]:
                tc_disp = tc if tc is not None else "∅"
                gc_disp = gc if gc is not None else "∅"
                mark = "✓" if ok else "x"
                details_lines.append(f"{i:02d}  |   {tc_disp}      |  {gc_disp}  | {mark}")
            self._set_details("\n".join(details_lines))
        else:
            self._set_details("")

        # Global score
        self.rounds_done += 1
        self.score_sum += float(result["accuracy"])
        self._update_progress()

        total = int(self.rounds_total.get())
        if self.rounds_done >= total:
            self.finish_game(auto=True)
        else:
            self.new_round()

        self._disable_buttons(False)

    def finish_game(self, auto: bool = False):
        total = int(self.rounds_total.get())
        if self.rounds_done == 0:
            messagebox.showinfo("Resultado", "Aún no has completado ninguna ronda.")
            return
        avg = self.score_sum / self.rounds_done
        title = "Rondas completadas" if auto else "Finalizado"
        messagebox.showinfo(title, f"Rondas: {self.rounds_done}/{total}\nMedia final: {avg:.1f}%")


# ==========================
# Main
# ==========================
if __name__ == "__main__":
    try:
        app = SpellingBeeApp(words_file=WORDS_FILE)
        app.mainloop()
    except Exception as e:
        try:
            messagebox.showerror("Error al iniciar", str(e))
        except Exception:
            print("Error al iniciar:", e)
