# Spelling Bee (Tkinter + micrófono) — Modo letras

Aplicación en Python con interfaz gráfica (Tkinter) para practicar *spelling* en inglés:
- Carga una lista de palabras desde `words.txt`
- Selecciona una palabra al azar por ronda
- El alumno **deletrea por micrófono** (modo letras)
- La app transcribe (ASR), normaliza a letras y calcula **accuracy (%)**
- Soporta **modo examen** configurable con `.env`

## 1) Estructura del proyecto

```

spelling_bee/
├── spelling_bee_tk.py
├── words.txt
├── requirements.txt
└── .env

```

Ejemplo de `words.txt` (una palabra por línea):

```

apple
computer
development
language
beautiful

````

## 2) Instalación

Crea un entorno virtual (recomendado):

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
````

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### requirements.txt (sugerido)

```txt
SpeechRecognition==3.10.0
pyaudio==0.2.14
python-dotenv==1.0.1
```

> Nota: `pyaudio` puede fallar al instalar en Windows.
> Si te ocurre, alternativa recomendada: versión con `sounddevice` (pídemela y te la paso).

## 3) Configuración con `.env`

Crea un fichero `.env` junto al script:

```env
# Archivo de palabras
WORDS_FILE=words.txt

# Rondas por defecto
ROUNDS_DEFAULT=10

# Modo examen: true oculta la palabra (muestra pista), false la muestra
EXAM_MODE=true

# Pista si EXAM_MODE=true
# none | first_letter | length | both
HINT_MODE=both

# Audio / tiempos
LISTEN_TIME_LIMIT=6
AMBIENT_DURATION=0.6

# UI
WINDOW_WIDTH=760
WINDOW_HEIGHT=520

# Mostrar detalle posición a posición
SHOW_DETAILS=true

# Scoring
SCORING_MODE=position
PENALIZE_EXTRA=true
```

### Significado rápido

* **EXAM_MODE=true**: no muestra la palabra, muestra pista según `HINT_MODE`
* **LISTEN_TIME_LIMIT**: segundos máximos de escucha por intento
* **AMBIENT_DURATION**: ajuste de ruido ambiente (sube si el aula es ruidosa)
* **SHOW_DETAILS**: muestra o no el detalle por posiciones

## 4) Ejecutar

```bash
python spelling_bee_tk.py
```

## 5) Uso

* Pulsa **🎙️ Escuchar** y deletrea la palabra letra a letra:

  * Ejemplo: “A P P L E”
  * También acepta: “double u”, “zee/zed”, etc.
* Botón **Repetir (no cuenta)**: permite practicar sin que sume ronda/nota.

## 6) Consejos de uso en clase

* Si el reconocimiento confunde letras, prueba:

  * aumentar `LISTEN_TIME_LIMIT` (p.ej. 8)
  * aumentar `AMBIENT_DURATION` (p.ej. 1.0)
  * pedir al alumno que vocalice y deje micro-pauses entre letras

## 7) Roadmap (si lo quieres)

Mejoras fáciles:

* Exportar resultados a CSV (palabra, intento, accuracy, timestamp)
* Texto a voz (TTS): pronunciar la palabra o la pista
* Backend offline (Vosk) para no depender de internet
* Modo “varias oportunidades” por palabra
