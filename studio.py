import atexit
import io
import os
import datetime
import random
import zipfile
import numpy as np
import gradio as gr
import soundfile as sf

from core.transcriber import Transcriber
from core.model_registry import ModelRegistry
from core.text_processing import (
    clean_text_fn, split_text_into_chunks, parse_pause_tags,
)
from core.audio_utils import (
    concat_audio, generate_silence, time_stretch, save_audio,
)
from core.postprocessing import PostProcessor

# --- Paths ---
VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(VOICES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Globals ---
transcriber = Transcriber()
registry = ModelRegistry()
postprocessor = PostProcessor()

EVENT_TAGS = [
    "[clear throat]", "[sigh]", "[shush]", "[cough]", "[groan]",
    "[sniff]", "[gasp]", "[chuckle]", "[laugh]",
]

CUSTOM_CSS = """
.tag-container {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    margin-top: 5px !important;
    margin-bottom: 10px !important;
    border: none !important;
    background: transparent !important;
}
.tag-btn {
    min-width: fit-content !important;
    width: auto !important;
    height: 32px !important;
    font-size: 13px !important;
    background: #eef2ff !important;
    border: 1px solid #c7d2fe !important;
    color: #3730a3 !important;
    border-radius: 6px !important;
    padding: 0 12px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}
.tag-btn:hover {
    background: #e0e7ff !important;
    border-color: #a5b4fc !important;
    transform: translateY(-1px) !important;
}
"""

# ─── Voice helpers ───────────────────────────────────────────────


def list_voices() -> list[str]:
    """Scan voices/ for .wav files, return sorted list of filenames."""
    if not os.path.isdir(VOICES_DIR):
        return []
    return sorted(
        f for f in os.listdir(VOICES_DIR)
        if f.lower().endswith(".wav")
    )


def get_transcript(voice_name: str) -> str | None:
    """Read the .txt transcript for a voice, or return None."""
    txt_path = os.path.join(
        VOICES_DIR,
        os.path.splitext(voice_name)[0] + ".txt",
    )
    if os.path.isfile(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None


def save_transcript(voice_name: str, text: str):
    """Write the .txt transcript for a voice."""
    txt_path = os.path.join(
        VOICES_DIR,
        os.path.splitext(voice_name)[0] + ".txt",
    )
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)


def voice_path(voice_name: str) -> str:
    """Return full path to a voice .wav file."""
    return os.path.join(VOICES_DIR, voice_name)


def list_output_files() -> list[str]:
    """List audio files in output/ sorted by modification time (newest first)."""
    if not os.path.isdir(OUTPUT_DIR):
        return []
    files = []
    for f in os.listdir(OUTPUT_DIR):
        if f.lower().endswith((".wav", ".mp3", ".flac")):
            full = os.path.join(OUTPUT_DIR, f)
            files.append((os.path.getmtime(full), f))
    files.sort(reverse=True)
    return [f for _, f in files]


# ─── .txt file upload ───────────────────────────────────────────


def load_txt_files(files):
    """Load uploaded .txt files, return (preview, status, [(name, text)...])."""
    if not files:
        return "", "No files uploaded.", []
    texts = []
    filenames = []
    for file in files:
        filepath = file if isinstance(file, str) else file.name
        if not filepath.endswith(".txt"):
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                texts.append(content)
                filenames.append(os.path.basename(filepath))
        except Exception as e:
            print(f"Error reading {file}: {e}")
    if not texts:
        return "", "No valid .txt files found.", []
    if len(texts) == 1:
        preview = texts[0]
    else:
        preview = f"{len(texts)} files loaded. Preview of first:\n\n{texts[0][:500]}..."
    status = f"Loaded {len(texts)} file(s): {', '.join(filenames)}"
    return preview, status, list(zip(filenames, texts))


# ─── Seed helper ─────────────────────────────────────────────────


def set_seed(seed: int):
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


# ─── Generation ──────────────────────────────────────────────────


def generate(
    text: str,
    txt_files,
    separate_files: bool,
    selected_voice: str,
    transcript_text: str,
    chunk_strategy: str,
    clean_whitespace: bool,
    move_punctuation: bool,
    replace_dashes: bool,
    normalize_nums: bool,
    expand_abbrev: bool,
    enable_pause_tags: bool,
    playback_speed: float,
    output_format: str,
    denoise: bool,
    super_resolution: bool,
    param_values: dict,
    loaded_texts: list,
    cleanup_interval: int,
    progress: gr.Progress | None = None,
):
    """Main generation pipeline."""
    if not registry.active_model:
        return None, "No model loaded. Select a model first.", gr.Dropdown(choices=[]), None, []

    if not selected_voice:
        return None, "No voice selected.", gr.Dropdown(choices=[]), None, []

    config = registry.get_config(registry.active_model)

    # Warn if model needs transcript but none provided
    if config.get("needs_transcript", False) and not transcript_text:
        x_vec = param_values.get("x_vector_only_mode", False)
        if not x_vec:
            return None, "This model requires a voice transcript. Transcribe or enter one, or enable X-Vector only mode.", gr.Dropdown(choices=[]), None, []

    # ── Gather texts to process ──
    texts_to_process = []
    filenames = []
    if loaded_texts:
        if separate_files:
            for filename, file_text in loaded_texts:
                texts_to_process.append(file_text)
                filenames.append(filename.replace(".txt", ""))
        else:
            texts_to_process.append("\n\n".join(t for _, t in loaded_texts))
            filenames.append("combined")
    elif text.strip():
        texts_to_process.append(text)
        filenames.append("manual_input")
    else:
        return None, "No text provided.", gr.Dropdown(choices=[]), None, []

    # ── Seed ──
    seed = int(param_values.get("seed", 0))
    if seed == 0:
        seed = random.randint(1, 1_000_000)
    set_seed(seed)

    sample_rate = config.get("sample_rate", 24000)
    all_generated_files = []
    status_messages = []
    chunks_since_cleanup = 0
    vpath = voice_path(selected_voice) if selected_voice else None

    for text_content, base_filename in zip(texts_to_process, filenames):
        # ── Pause tags ──
        if enable_pause_tags and config.get("supports_event_tags", False):
            segments = parse_pause_tags(text_content)
        else:
            segments = [{"type": "text", "content": text_content}]

        audio_segments = []

        for segment in segments:
            if segment["type"] == "pause":
                audio_segments.append(
                    generate_silence(segment["duration"], sample_rate)
                )
                continue

            chunk_text = segment["content"]

            # ── Text preprocessing (only if model declares it) ──
            if config.get("needs_text_preprocessing", False):
                chunk_text = clean_text_fn(
                    chunk_text,
                    clean_whitespace=clean_whitespace,
                    move_punctuation=move_punctuation,
                    replace_dashes=replace_dashes,
                    normalize_nums=normalize_nums,
                    expand_abbrev=expand_abbrev,
                )

            chunks = split_text_into_chunks(chunk_text, strategy=chunk_strategy)
            total_chunks = len(chunks)

            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                if progress is not None:
                    progress((i + 1) / total_chunks, desc=f"Generating chunk {i + 1}/{total_chunks}")
                try:
                    wav_bytes = registry.generate(
                        text=chunk,
                        voice_path=vpath,
                        voice_transcript=transcript_text or "",
                        params=param_values,
                    )
                    audio_arr, _ = sf.read(io.BytesIO(wav_bytes), dtype="float32")
                    audio_segments.append(audio_arr)

                    chunks_since_cleanup += 1
                    if chunks_since_cleanup >= cleanup_interval:
                        import torch, gc
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        chunks_since_cleanup = 0
                except Exception as e:
                    print(f"Generation error on chunk {i}: {e}")
                    status_messages.append(f"Error on chunk {i}: {e}")

        if not audio_segments:
            continue

        full_audio = concat_audio(audio_segments)

        # ── Save raw WAV ──
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{base_filename}_{timestamp}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        sf.write(output_path, full_audio, sample_rate, subtype="PCM_16")

        # ── Post-processing ──
        if progress is not None and (denoise or super_resolution):
            progress(0, desc="Post-processing...")
        output_path = postprocessor.process(
            output_path, denoise=denoise, super_resolution=super_resolution,
        )

        # ── Speed adjustment + format conversion ──
        if playback_speed != 1.0 or output_format != "wav":
            import librosa as lr
            y, file_sr = lr.load(output_path, sr=sample_rate)
            if playback_speed != 1.0:
                y = time_stretch(y, rate=playback_speed)
            final_path = save_audio(y, output_path, file_sr, output_format)
            if output_format != "wav" and os.path.exists(output_path):
                os.remove(output_path)
            output_path = final_path

        all_generated_files.append((os.path.basename(output_path), output_path))
        status_messages.append(f"{base_filename} generated.")

    if not all_generated_files:
        return None, "Generation failed.", gr.Dropdown(choices=[]), None, []

    # ── Session ZIP ──
    dropdown_choices = [f"{i+1}. {f[0]}" for i, f in enumerate(all_generated_files)]
    zip_path = os.path.join(OUTPUT_DIR, f"session_{random.randint(1000, 9999)}.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for fname, fpath in all_generated_files:
            zf.write(fpath, fname)

    return (
        all_generated_files[0][1],
        "\n".join(status_messages),
        gr.Dropdown(choices=dropdown_choices, value=dropdown_choices[0]),
        zip_path,
        all_generated_files,
    )


# ─── UI Builder ──────────────────────────────────────────────────


def build_app():
    registry.discover()
    model_ids = list(registry.models.keys())
    model_names = {mid: registry.models[mid]["name"] for mid in model_ids}

    with gr.Blocks(css=CUSTOM_CSS, title="Balauru's TTS Studio") as demo:
        # ── State ──
        loaded_texts_state = gr.State([])
        generated_files_state = gr.State([])

        gr.Markdown("# Balauru's TTS Studio")

        # ── Top bar: model selector + status ──
        with gr.Row():
            model_dropdown = gr.Dropdown(
                label="Model",
                choices=[(model_names[mid], mid) for mid in model_ids],
                value=None,
                scale=3,
            )
            model_status = gr.Textbox(
                value="No model loaded",
                label="Status",
                interactive=False,
                scale=2,
            )

        with gr.Row():
            # ══════════ LEFT COLUMN ══════════
            with gr.Column(scale=1):
                # ── Voice selector ──
                with gr.Group():
                    gr.Markdown("### Voice")
                    voice_dropdown = gr.Dropdown(
                        label="Select Voice",
                        choices=list_voices(),
                    )
                    voice_refresh_btn = gr.Button("Refresh", size="sm")
                    voice_preview = gr.Audio(
                        label="Preview", interactive=False,
                    )
                    transcript_box = gr.Textbox(
                        label="Transcript (for Qwen3-TTS)",
                        lines=3,
                        placeholder="Transcript of the voice sample...",
                    )
                    with gr.Row():
                        transcribe_btn = gr.Button("Transcribe", size="sm")
                        save_transcript_btn = gr.Button("Save Transcript", size="sm")

                # ── Text input ──
                gr.Markdown("### Text Input")
                text_input = gr.Textbox(
                    label="Text to synthesize",
                    lines=6,
                    placeholder="Enter text here...",
                )

                # ── Event tags (conditionally visible) ──
                event_tags_group = gr.Group(visible=False)
                with event_tags_group:
                    gr.Markdown("**Sound Effects**")
                    tag_container = gr.Group(elem_classes="tag-container")
                    with tag_container:
                        tag_buttons = []
                        for tag in EVENT_TAGS:
                            btn = gr.Button(tag, elem_classes="tag-btn", size="sm")
                            tag_buttons.append((btn, tag))

                # ── .txt file upload ──
                with gr.Accordion("Upload .txt Files", open=False):
                    txt_files_input = gr.File(
                        label="Upload .txt files",
                        file_count="multiple",
                        file_types=[".txt"],
                    )
                    with gr.Row():
                        load_txt_btn = gr.Button("Load Files")
                        separate_files_chk = gr.Checkbox(
                            label="Separate files", value=False,
                        )
                    txt_status = gr.Markdown()

                # ── Text processing (conditionally visible) ──
                text_proc_group = gr.Group(visible=False)
                with text_proc_group:
                    gr.Markdown("**Text Processing**")
                    clean_ws_chk = gr.Checkbox(label="Normalize Whitespace", value=True)
                    move_punct_chk = gr.Checkbox(label="Move Punctuation", value=True)
                    replace_dashes_chk = gr.Checkbox(label="Replace Dashes", value=True)
                    normalize_nums_chk = gr.Checkbox(label="Normalize Numbers", value=False)
                    expand_abbrev_chk = gr.Checkbox(label="Expand Abbreviations", value=False)

                enable_pause_chk = gr.Checkbox(label="Enable [pause:Xs] Tags", value=True)
                chunk_dropdown = gr.Dropdown(
                    label="Chunking Strategy",
                    choices=["Sentence Batching (<300 chars)"],
                    value="Sentence Batching (<300 chars)",
                )

                run_btn = gr.Button("Generate Audio", variant="primary", size="lg")
                generation_status = gr.Markdown()

            # ══════════ RIGHT COLUMN ══════════
            with gr.Column(scale=1):
                audio_output = gr.Audio(label="Audio Preview", interactive=False)

                with gr.Accordion("Session Output", open=True):
                    audio_selector = gr.Dropdown(label="Current Session Files", choices=[])
                    download_zip = gr.File(label="Download Session ZIP")

                with gr.Accordion("Output History", open=False):
                    history_dropdown = gr.Dropdown(
                        label="Previous Outputs",
                        choices=list_output_files(),
                    )
                    history_refresh_btn = gr.Button("Refresh", size="sm")

                with gr.Accordion("Output Format", open=True):
                    output_format = gr.Radio(
                        ["wav", "mp3", "flac"], value="wav", label="Format",
                    )
                    playback_speed = gr.Slider(
                        0.5, 2.0, value=1.0, step=0.05, label="Playback Speed",
                    )

                with gr.Accordion("Post-Processing", open=True):
                    denoise_chk = gr.Checkbox(label="Denoise (MossFormer2)", value=False)
                    super_res_chk = gr.Checkbox(label="Super-Resolution (48kHz)", value=False)

                # ── Dynamic Parameters ──
                # Pre-build parameter UIs for ALL discovered models.
                # Only the active model's column is visible at a time.
                with gr.Accordion("Parameters", open=True):
                    param_columns = {}       # model_id → gr.Column
                    param_components = {}    # model_id → {param_name: component}

                    for mid in model_ids:
                        config = registry.models[mid]
                        visible = False
                        col = gr.Column(visible=visible)
                        param_columns[mid] = col
                        param_components[mid] = {}
                        with col:
                            for p in config.get("parameters", []):
                                name = p["name"]
                                ptype = p["type"]
                                label = p.get("label", name.replace("_", " ").title())
                                if ptype == "slider":
                                    param_components[mid][name] = gr.Slider(
                                        minimum=p["min"], maximum=p["max"],
                                        value=p["default"], step=p.get("step", 0.01),
                                        label=label,
                                    )
                                elif ptype == "number":
                                    param_components[mid][name] = gr.Number(
                                        value=p["default"], label=label,
                                    )
                                elif ptype == "checkbox":
                                    param_components[mid][name] = gr.Checkbox(
                                        value=p["default"], label=label,
                                    )
                                elif ptype == "dropdown":
                                    param_components[mid][name] = gr.Dropdown(
                                        choices=p.get("choices", []),
                                        value=p["default"], label=label,
                                    )

                    # If no models discovered, show placeholder
                    if not model_ids:
                        gr.Markdown("No models discovered.")

                cleanup_interval = gr.Slider(
                    5, 200, value=25, step=5,
                    label="VRAM Cleanup Interval (chunks)",
                )

        # ── Build flat list of all param components for generate inputs ──
        # Maps (index in flat list) → (model_id, param_name)
        param_flat_list = []
        param_flat_map = []
        for mid in model_ids:
            for pname, comp in param_components[mid].items():
                param_flat_list.append(comp)
                param_flat_map.append((mid, pname))

        # ─── Event wiring ──────────────────────────────────────

        # Voice selection
        def on_voice_select(voice_name):
            if not voice_name:
                return None, ""
            path = voice_path(voice_name)
            transcript = get_transcript(voice_name) or ""
            return path, transcript

        voice_dropdown.change(
            on_voice_select,
            inputs=[voice_dropdown],
            outputs=[voice_preview, transcript_box],
        )

        voice_refresh_btn.click(
            lambda: gr.Dropdown(choices=list_voices()),
            outputs=[voice_dropdown],
        )

        # Transcribe button
        def on_transcribe(voice_name):
            if not voice_name:
                return "No voice selected."
            path = voice_path(voice_name)
            try:
                text = transcriber.transcribe(path)
                save_transcript(voice_name, text)
                return text
            except Exception as e:
                return f"Transcription error: {e}"

        transcribe_btn.click(
            on_transcribe,
            inputs=[voice_dropdown],
            outputs=[transcript_box],
        )

        # Save transcript button
        def on_save_transcript(voice_name, text):
            if not voice_name:
                return "No voice selected."
            save_transcript(voice_name, text)
            return "Transcript saved."

        save_transcript_btn.click(
            on_save_transcript,
            inputs=[voice_dropdown, transcript_box],
            outputs=[generation_status],
        )

        # Event tag insertion
        for btn, tag in tag_buttons:
            btn.click(
                lambda cur, t=tag: cur + " " + t,
                inputs=[text_input],
                outputs=[text_input],
            )

        # .txt file loading
        load_txt_btn.click(
            load_txt_files,
            inputs=[txt_files_input],
            outputs=[text_input, txt_status, loaded_texts_state],
        )

        # ── Model switching ──
        # Outputs: model_status, event_tags_group, text_proc_group,
        #          chunk_dropdown, transcript_box,
        #          + one visibility update per param_column
        def on_model_switch(model_id):
            # Base outputs
            if not model_id:
                base = [
                    "No model selected.",
                    gr.Group(visible=False),
                    gr.Group(visible=False),
                    gr.Dropdown(choices=[]),
                    gr.Textbox(visible=True),
                ]
                col_updates = [gr.Column(visible=False) for _ in model_ids]
                return base + col_updates

            try:
                registry.activate(model_id)
                config = registry.get_config(model_id)

                show_tags = config.get("supports_event_tags", False)
                show_preproc = config.get("needs_text_preprocessing", False)
                chunks = config.get("chunk_strategies", ["No Split"])

                base = [
                    f"Ready: {config['name']}",
                    gr.Group(visible=show_tags),
                    gr.Group(visible=show_preproc),
                    gr.Dropdown(choices=chunks, value=chunks[0]),
                    gr.Textbox(visible=config.get("needs_transcript", False)),
                ]
                # Show only the active model's param column
                col_updates = [
                    gr.Column(visible=(mid == model_id)) for mid in model_ids
                ]
                return base + col_updates

            except Exception as e:
                base = [
                    f"Error: {e}",
                    gr.Group(visible=False),
                    gr.Group(visible=False),
                    gr.Dropdown(choices=[]),
                    gr.Textbox(visible=True),
                ]
                col_updates = [gr.Column(visible=False) for _ in model_ids]
                return base + col_updates

        model_switch_outputs = [
            model_status,
            event_tags_group,
            text_proc_group,
            chunk_dropdown,
            transcript_box,
        ] + [param_columns[mid] for mid in model_ids]

        model_dropdown.change(
            on_model_switch,
            inputs=[model_dropdown],
            outputs=model_switch_outputs,
        )

        # File preview selector (current session)
        audio_selector.change(
            lambda s, f: f[int(s.split(".")[0]) - 1][1] if s and f else None,
            inputs=[audio_selector, generated_files_state],
            outputs=[audio_output],
        )

        # Output history
        def on_history_select(filename):
            if not filename:
                return None
            return os.path.join(OUTPUT_DIR, filename)

        history_dropdown.change(
            on_history_select,
            inputs=[history_dropdown],
            outputs=[audio_output],
        )

        history_refresh_btn.click(
            lambda: gr.Dropdown(choices=list_output_files()),
            outputs=[history_dropdown],
        )

        # ── Generate button ──
        # Fixed inputs + all dynamic param components (flat list)
        fixed_inputs = [
            text_input, txt_files_input, separate_files_chk,
            voice_dropdown, transcript_box,
            chunk_dropdown,
            clean_ws_chk, move_punct_chk, replace_dashes_chk,
            normalize_nums_chk, expand_abbrev_chk, enable_pause_chk,
            playback_speed, output_format, denoise_chk, super_res_chk,
            loaded_texts_state, cleanup_interval,
        ]
        all_inputs = fixed_inputs + param_flat_list

        def on_generate(progress=gr.Progress(), *args):
            n_fixed = len(fixed_inputs)
            fixed_vals = args[:n_fixed]
            param_vals_raw = args[n_fixed:]

            # Reconstruct param dict for the active model
            active = registry.active_model
            params = {}
            for (mid, pname), val in zip(param_flat_map, param_vals_raw):
                if mid == active:
                    params[pname] = val

            (text, txt_files, separate_files, selected_voice, transcript_text,
             chunk_strategy, clean_ws, move_punct, replace_dashes, normalize_n,
             expand_abb, enable_pause, playback_spd, out_fmt, dn, sr,
             loaded_texts, cleanup_int) = fixed_vals

            return generate(
                text=text,
                txt_files=txt_files,
                separate_files=separate_files,
                selected_voice=selected_voice,
                transcript_text=transcript_text,
                chunk_strategy=chunk_strategy,
                clean_whitespace=clean_ws,
                move_punctuation=move_punct,
                replace_dashes=replace_dashes,
                normalize_nums=normalize_n,
                expand_abbrev=expand_abb,
                enable_pause_tags=enable_pause,
                playback_speed=playback_spd,
                output_format=out_fmt,
                denoise=dn,
                super_resolution=sr,
                param_values=params,
                loaded_texts=loaded_texts,
                cleanup_interval=cleanup_int,
                progress=progress,
            )

        run_btn.click(
            on_generate,
            inputs=all_inputs,
            outputs=[
                audio_output, generation_status, audio_selector,
                download_zip, generated_files_state,
            ],
        )

    return demo


if __name__ == "__main__":
    atexit.register(registry.shutdown)
    app = build_app()
    app.launch()
