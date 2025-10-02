from __future__ import annotations
import os, uuid, logging, traceback
from datetime import datetime
from fractions import Fraction
from flask import Flask, render_template, request, send_from_directory, abort
from werkzeug.utils import secure_filename
import ffmpeg
from PIL import Image
import imagehash

app = Flask(__name__, static_url_path='/static', template_folder='templates')

UPLOAD_DIR  = os.environ.get("UPLOAD_DIR",  "uploads")
OUTPUT_DIR  = os.environ.get("OUTPUT_DIR",  "outputs")
LOG_LEVEL   = os.environ.get("LOG_LEVEL",   "INFO").upper()
MAX_SIZE_GB = float(os.environ.get("MAX_SIZE_GB", "2"))
ALLOWED_EXTS = {'.mp4', '.mov', '.mkv', '.avi', '.m4v', '.webm'}

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.config["UPLOAD_FOLDER"]      = UPLOAD_DIR
app.config["OUTPUT_FOLDER"]      = OUTPUT_DIR
app.config["MAX_CONTENT_LENGTH"] = int(MAX_SIZE_GB * 1024 * 1024 * 1024)

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(asctime)s | %(levelname)s | %(message)s")

def make_error_stats(message: str) -> dict:
    return {
        "filename": None,
        "video_info": None,
        "scenes_detected": 0,
        "unique_scenes": 0,
        "removed_scenes": 0,
        "compressed_filename": None,
        "compression_stats": None,
        "scenes_timeline": [],
        "unique_scenes_details": [],
        "removed_scenes_details": [],
        "error": message
    }

def probe_safe(filepath: str) -> dict:
    try:
        return ffmpeg.probe(filepath)
    except ffmpeg.Error as e:
        err = e.stderr.decode('utf-8', errors='ignore') if getattr(e, "stderr", None) else str(e)
        logging.error(f"[ffprobe] error for {filepath}:\n{err}")
        raise RuntimeError(f"ffprobe error: {err}") from e
    except Exception as e:
        logging.error(f"[ffprobe] generic failure: {e}")
        raise

def get_video_info(filepath: str) -> dict:
    probe = probe_safe(filepath)
    fmt = probe.get("format", {}) or {}
    streams = probe.get("streams", []) or []
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        raise RuntimeError("Видео-поток не найден")
    fps = 25.0
    try:
        if v.get("r_frame_rate") and v["r_frame_rate"] != "0/0":
            num, den = v["r_frame_rate"].split("/")
            fps = float(Fraction(int(num), int(den)))
    except Exception:
        pass
    duration = None
    if v.get("duration") is not None:
        try:
            duration = float(v["duration"])
        except Exception:
            duration = None
    if duration is None:
        try:
            duration = float(fmt.get("duration", 0.0))
        except Exception:
            duration = 0.0
    width  = int(v.get("width", 0) or 0)
    height = int(v.get("height", 0) or 0)
    try:
        bitrate = int(v.get("bit_rate", 0) or 0)
    except Exception:
        bitrate = 0
    info = {"fps": fps, "duration": duration, "width": width, "height": height, "bitrate": bitrate}
    logging.info(f"[probe] fps={fps:.3f}, dur={duration:.3f}s, {width}x{height}, br={bitrate}")
    return info

def merge_intervals(intervals, tol=1e-3):
    if not intervals: return []
    intervals = sorted(intervals, key=lambda x: (x[0], x[1]))
    merged = []
    cs, ce = intervals[0]
    for s, e in intervals[1:]:
        if s <= ce + tol:
            ce = max(ce, e)
        else:
            merged.append((cs, ce)); cs, ce = s, e
    merged.append((cs, ce))
    return merged

def complement_intervals(full_start, full_end, drops):
    if not drops:
        return [(full_start, full_end)]
    drops = merge_intervals(drops)
    kept, cursor = [], full_start
    for ds, de in drops:
        if ds > cursor:
            kept.append((cursor, min(ds, full_end)))
        cursor = max(cursor, de)
        if cursor >= full_end:
            break
    if cursor < full_end:
        kept.append((cursor, full_end))
    return [(max(full_start, s), min(full_end, e)) for s, e in kept if e - s > 0]

def merge_adjacent_scenes(scenes, join_gap=0.02):
    if not scenes: return scenes
    scenes = sorted(scenes, key=lambda s: (s["start"], s["end"]))
    merged, cur = [], scenes[0].copy()
    for s in scenes[1:]:
        if s["start"] - cur["end"] <= join_gap:
            cur["end"] = max(cur["end"], s["end"])
            cur["duration"] = round(cur["end"] - cur["start"], 3)
        else:
            merged.append(cur); cur = s.copy()
    merged.append(cur)
    return merged

def iter_frames_raw(filepath: str, target_fps: int, scaled_width: int):
    info = get_video_info(filepath)
    if info["width"] and info["height"]:
        scaled_h = int(round(scaled_width * (info["height"] / info["width"])))
    else:
        scaled_h = int(scaled_width * 9 / 16)
    scaled_h = max(1, scaled_h)
    logging.info(f"[frames] pipe start: fps={target_fps}, size={scaled_width}x{scaled_h}")
    process = (
        ffmpeg
        .input(filepath)
        .filter('fps', fps=target_fps, round='up')
        .filter('scale', scaled_width, scaled_h)
        .output('pipe:', format='rawvideo', pix_fmt='rgb24')
        .global_args('-hide_banner', '-nostdin', '-loglevel', 'error')
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )
    frame_size = scaled_width * scaled_h * 3
    frame_idx = 0
    try:
        while True:
            chunk = process.stdout.read(frame_size)
            if not chunk or len(chunk) < frame_size:
                break
            try:
                img = Image.frombytes('RGB', (scaled_width, scaled_h), chunk)
            except Exception as e:
                logging.warning(f"[frames] build image failed idx={frame_idx}: {e}")
                break
            yield frame_idx, img
            frame_idx += 1
    finally:
        try:
            err = process.stderr.read().decode('utf-8', errors='ignore')
            if err.strip():
                logging.error(f"[ffmpeg:rawvideo] stderr:\n{err}")
        except Exception:
            pass
        try:
            process.stdout.close()
        except Exception:
            pass
        process.wait()
    if frame_idx == 0:
        raise RuntimeError("Не удалось прочитать ни одного кадра. См. stderr FFmpeg выше.")

def detect_repetition_runs(
    filepath: str,
    run_sim_threshold=6,
    window_frames=9,
    flat_ratio=0.85,
    min_flat_sec=0.7,
    keep_margin_sec=0.25,
    min_keep_duration=0.35,
    merge_gap=0.05,
    scaled_width=320,
    analysis_fps=None,
    min_total_drop_ratio=0.05,
    min_total_drop_sec=1.0,
    max_relax_steps=2
):
    logging.info("[detect] start")
    info = get_video_info(filepath)
    fps0 = info["fps"] if info["fps"] > 0 else 25.0
    fps_an = int(round(analysis_fps or min(fps0, 15)))
    total_dur = info["duration"]
    hashes, n_frames = [], 0
    for idx, img in iter_frames_raw(filepath, target_fps=fps_an, scaled_width=scaled_width):
        try:
            h = imagehash.phash(img)
        except Exception as e:
            logging.warning(f"[detect] phash failed at {idx}: {e}")
            h = None
        hashes.append(h); n_frames += 1
    logging.info(f"[detect] frames hashed: {n_frames}")
    if n_frames <= 2 or total_dur <= 0:
        scenes = [{"start": 0.0, "end": round(total_dur, 3), "duration": round(total_dur, 3)}]
        removed_runs = []
        logging.info("[detect] not enough frames or unknown duration → keep whole video")
        return scenes, removed_runs, info
    def f2t(fr): return min(total_dur, fr / max(1, fps_an))
    base_thr, base_min_flat = run_sim_threshold, min_flat_sec
    drops_final, removed_runs = None, []
    for step in range(max_relax_steps + 1):
        thr = base_thr + step
        min_flat = max(0.35, base_min_flat - 0.2 * step)
        logging.info(f"[detect] relax step={step} thr={thr} min_flat={min_flat}")
        diffs = []
        for i in range(1, n_frames):
            a, b = hashes[i-1], hashes[i]
            diffs.append(999 if (a is None or b is None) else abs(a - b))
        flat = [False] * n_frames
        half = max(1, window_frames // 2)
        for i in range(n_frames):
            dL, dR = max(0, i-half), min(len(diffs)-1, i+half-1)
            if dR < dL: continue
            window = diffs[dL:dR+1]
            small = sum(1 for d in window if d <= thr)
            flat[i] = (small / max(1, len(window))) >= flat_ratio
        runs, i = [], 0
        while i < n_frames:
            if not flat[i]:
                i += 1; continue
            j = i
            while j+1 < n_frames and flat[j+1]: j += 1
            s, e = f2t(i), f2t(j+1)
            if e - s >= min_flat: runs.append((s, e))
            i = j + 1
        logging.info(f"[detect] flat runs: {len(runs)}")
        margin = keep_margin_sec
        drops, removed_runs = [], []
        for rs, re in runs:
            if re - rs <= 2 * margin: continue
            inner_s, inner_e = rs + margin, re - margin
            if inner_e - inner_s > 0:
                drops.append((inner_s, inner_e))
                removed_runs.append({"start": round(inner_s, 3), "end": round(inner_e, 3), "duration": round(inner_e - inner_s, 3)})
        drops = merge_intervals(drops, tol=1e-3)
        total_drop = sum(e - s for s, e in drops)
        logging.info(f"[detect] drops={len(drops)} total_drop={round(total_drop,3)}s")
        if total_drop >= max(min_total_drop_ratio * total_dur, min_total_drop_sec) or step == max_relax_steps:
            drops_final = drops
            break
    kept = complement_intervals(0.0, total_dur, drops_final)
    kept = [(s, e) for (s, e) in kept if (e - s) >= min_keep_duration]
    kept = merge_intervals(kept, tol=merge_gap)
    scenes = [{"start": round(s, 3), "end": round(e, 3), "duration": round(e - s, 3)} for (s, e) in kept]
    scenes = merge_adjacent_scenes(scenes, join_gap=merge_gap)
    logging.info(f"[detect] kept={len(scenes)} removed_segments={len(removed_runs)}")
    return scenes, removed_runs, info

def create_compressed_video_filter(
    filepath: str, scenes: list[dict], output_filename: str,
    target_fps: int | None = None, crf=22, preset='fast',
    audio_sr=48000, audio_ch=2,
    fade_duration=0.0,
    scene_padding=0.12,
    transition: str = "crossfade",
    xfade_duration: float = 0.12
) -> str | None:
    if not scenes:
        logging.error("[build] No scenes to concat")
        return None
    probe = probe_safe(filepath)
    has_audio = any(s.get("codec_type") == "audio" for s in probe.get("streams", []))
    logging.info(f"[build] has_audio={has_audio}")
    info = get_video_info(filepath)
    dur_total = float(info["duration"])
    tgt_fps = int(round(target_fps or (info["fps"] or 25)))
    out_path = os.path.join(app.config["OUTPUT_FOLDER"], output_filename)
    try:
        if os.path.exists(out_path): os.remove(out_path)
    except Exception:
        pass
    parts = []
    for s in scenes:
        start = round(max(0.0, s["start"] - scene_padding), 3)
        end   = round(min(dur_total, s["end"]   + scene_padding), 3)
        if end - start >= 0.02:
            parts.append((start, end))
    if not parts:
        logging.error("[build] All scenes too short after padding")
        return None
    logging.info(f"[build] parts={len(parts)}")
    out_kwargs = dict(vcodec='libx264', preset=preset, crf=crf, pix_fmt='yuv420p', movflags='+faststart', shortest=None)
    if has_audio:
        out_kwargs.update(dict(acodec='aac', ar=audio_sr, ac=audio_ch))
    iv = ffmpeg.input(filepath)
    if len(parts) == 1:
        start, end = parts[0]
        logging.info(f"[build] single {start}..{end}")
        v = iv.video.trim(start=start, end=end).setpts('PTS-STARTPTS').filter('fps', fps=tgt_fps)
        if has_audio:
            a = iv.audio.filter('atrim', start=start, end=end).filter('asetpts', 'PTS-STARTPTS')
            try:
                ffmpeg.output(v, a, out_path, **out_kwargs).global_args('-hide_banner', '-nostdin', '-fflags', '+genpts').run(overwrite_output=True)
            except Exception as e:
                logging.error(f"[build] single-part failed: {e}\n{traceback.format_exc()}")
                return None
        else:
            try:
                ffmpeg.output(v, out_path, **out_kwargs).global_args('-hide_banner', '-nostdin', '-fflags', '+genpts').run(overwrite_output=True)
            except Exception as e:
                logging.error(f"[build] single-part failed: {e}\n{traceback.format_exc()}")
                return None
        ok = os.path.exists(out_path) and os.path.getsize(out_path) > 0
        logging.info(f"[build] single-part ok={ok}")
        return out_path if ok else None
    v_clips, a_clips, durations = [], [], []
    for (start, end) in parts:
        seg_dur = max(0.0, end - start)
        durations.append(seg_dur)
        vv = iv.video.trim(start=start, end=end).setpts('PTS-STARTPTS').filter('fps', fps=tgt_fps)
        v_clips.append(vv)
        if has_audio:
            aa = iv.audio.filter('atrim', start=start, end=end).filter('asetpts', 'PTS-STARTPTS')
            a_clips.append(aa)
    if transition.lower() == "cut":
        try:
            if has_audio:
                interleaved = []
                for i in range(len(v_clips)):
                    interleaved += [v_clips[i], a_clips[i]]
                joined = ffmpeg.concat(*interleaved, v=1, a=1, n=len(v_clips)).node
                out_v, out_a = joined[0], joined[1]
                ffmpeg.output(out_v, out_a, out_path, **out_kwargs).global_args('-hide_banner', '-nostdin', '-fflags', '+genpts').run(overwrite_output=True)
            else:
                out_v = ffmpeg.concat(*v_clips, v=1, a=0, n=len(v_clips)).node[0]
                ffmpeg.output(out_v, out_path, **out_kwargs).global_args('-hide_banner', '-nostdin', '-fflags', '+genpts').run(overwrite_output=True)
        except Exception as e:
            logging.error(f"[build] concat (cut) failed: {e}\n{traceback.format_exc()}")
            return None
        ok = os.path.exists(out_path) and os.path.getsize(out_path) > 0
        logging.info(f"[build] concat-cut ok={ok}")
        return out_path if ok else None
    try:
        cur_v = v_clips[0]
        cur_a = a_clips[0] if has_audio else None
        cur_dur = durations[0]
        for idx in range(1, len(v_clips)):
            nxt_v = v_clips[idx]
            nxt_a = a_clips[idx] if has_audio else None
            nxt_dur = durations[idx]
            d = max(0.02, min(xfade_duration, cur_dur * 0.5, nxt_dur * 0.5))
            off = max(0.0, cur_dur - d)
            cur_v = ffmpeg.filter([cur_v, nxt_v], 'xfade', transition='fade', duration=d, offset=off)
            if has_audio:
                cur_a = ffmpeg.filter([cur_a, nxt_a], 'acrossfade', d=d)
            cur_dur = cur_dur + nxt_dur - d
        if has_audio:
            ffmpeg.output(cur_v, cur_a, out_path, **out_kwargs).global_args('-hide_banner', '-nostdin', '-fflags', '+genpts').run(overwrite_output=True)
        else:
            ffmpeg.output(cur_v, out_path, **out_kwargs).global_args('-hide_banner', '-nostdin', '-fflags', '+genpts').run(overwrite_output=True)
    except Exception as e:
        logging.error(f"[build] crossfade failed: {e}\n{traceback.format_exc()}")
        return None
    ok = os.path.exists(out_path) and os.path.getsize(out_path) > 0
    logging.info(f"[build] crossfade ok={ok}")
    return out_path if ok else None

def calculate_compression_stats(original_path: str, compressed_path: str, removed_runs: list[dict] | None):
    try:
        orig = get_video_info(original_path)
        comp = get_video_info(compressed_path)
        osz  = os.path.getsize(original_path)
        csz  = os.path.getsize(compressed_path)
        total_removed = round(sum(r.get("duration", 0.0) for r in (removed_runs or [])), 2)
        return {
            "original_duration": round(orig['duration'], 2),
            "compressed_duration": round(comp['duration'], 2),
            "original_size_mb": round(osz / (1024 * 1024), 2),
            "compressed_size_mb": round(csz / (1024 * 1024), 2),
            "size_reduction_percent": round((1 - csz / osz) * 100, 2) if osz > 0 else 0,
            "duration_reduction_percent": round((1 - comp['duration'] / orig['duration']) * 100, 2) if orig['duration'] > 0 else 0,
            "scenes_removed": len(removed_runs or []),
            "time_removed_seconds": total_removed,
            "time_saved_seconds": round(orig['duration'] - comp['duration'], 2)
        }
    except Exception as e:
        logging.error(f"[stats] error: {e}")
        return None

@app.before_request
def log_request_meta():
    logging.info(f"[request] {request.method} {request.path} | len={request.content_length} | ctype={request.content_type} | form={list(request.form.keys())} | files={list(request.files.keys())}")

@app.errorhandler(413)
def too_large(e):
    return render_template("result.html", stats=make_error_stats("Файл слишком большой (413)")), 413

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    if not request.content_type or "multipart/form-data" not in (request.content_type or "").lower():
        return render_template("result.html", stats=make_error_stats("Форма должна быть multipart/form-data"))
    if "file" not in request.files:
        return render_template("result.html", stats=make_error_stats('Поле файла не найдено (ожидается name="file")'))
    file = request.files["file"]
    if file.filename == "":
        return render_template("result.html", stats=make_error_stats("Файл не выбран"))
    orig_name = file.filename or "upload.bin"
    ext = os.path.splitext(orig_name)[1].lower()
    if ext not in ALLOWED_EXTS:
        return render_template("result.html", stats=make_error_stats(f"Недопустимый тип файла: {ext}"))
    uid = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = secure_filename(f"{timestamp}_{uid}{ext}")
    in_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    try:
        file.save(in_path)
        logging.info(f"[upload] saved: {in_path} ({os.path.getsize(in_path)} bytes)")
    except Exception as e:
        logging.error(f"[upload] save failed: {e}\n{traceback.format_exc()}")
        return render_template("result.html", stats=make_error_stats(f"Не удалось сохранить файл: {e}"))
    try:
        run_sim_threshold = int(request.form.get("run_sim_threshold", 6))
        window_frames     = int(request.form.get("window_frames", 9))
        flat_ratio        = float(request.form.get("flat_ratio", 0.85))
        min_flat_sec      = float(request.form.get("min_flat_sec", 0.7))
        keep_margin_sec   = float(request.form.get("keep_margin_sec", 0.25))
        min_keep_duration = float(request.form.get("min_keep_duration", 0.35))
        merge_gap         = float(request.form.get("merge_gap", 0.05))
        scaled_width      = int(request.form.get("scaled_width", 320))
        target_fps_in     = request.form.get("target_fps", "").strip()
        analysis_fps      = int(target_fps_in) if target_fps_in else None
        output_fps        = analysis_fps
        scene_padding     = float(request.form.get("scene_padding", 0.12))
        fade_duration     = float(request.form.get("fade_duration", 0.0))
        crf               = int(request.form.get("crf", 22))
        preset            = request.form.get("preset", "fast")
        transition        = (request.form.get("transition") or "crossfade").strip()
        xfade_duration    = float(request.form.get("xfade_duration", 0.12))
        logging.info("[params] "
                     f"thr={run_sim_threshold}, win={window_frames}, flat={flat_ratio}, "
                     f"min_flat={min_flat_sec}, margin={keep_margin_sec}, "
                     f"min_keep={min_keep_duration}, merge_gap={merge_gap}, "
                     f"scaled_w={scaled_width}, analysis_fps={analysis_fps}, output_fps={output_fps}, "
                     f"pad={scene_padding}, fade={fade_duration}, crf={crf}, preset={preset}, "
                     f"transition={transition}, xfade={xfade_duration}")
        scenes, removed_runs, video_info = detect_repetition_runs(
            in_path,
            run_sim_threshold=run_sim_threshold,
            window_frames=window_frames,
            flat_ratio=flat_ratio,
            min_flat_sec=min_flat_sec,
            keep_margin_sec=keep_margin_sec,
            min_keep_duration=min_keep_duration,
            merge_gap=merge_gap,
            scaled_width=scaled_width,
            analysis_fps=analysis_fps
        )
        out_name = f"compressed_{safe_name}"
        out_path = create_compressed_video_filter(
            in_path, scenes, out_name,
            target_fps=output_fps,
            crf=crf, preset=preset,
            fade_duration=fade_duration,
            scene_padding=scene_padding,
            transition=transition,
            xfade_duration=xfade_duration
        )
        compressed_filename = os.path.basename(out_path) if out_path else None
        compression_stats = None
        if out_path and os.path.exists(out_path):
            compression_stats = calculate_compression_stats(in_path, out_path, removed_runs)
        stats = {
            "filename": safe_name,
            "video_info": video_info,
            "scenes_detected": len(scenes),
            "unique_scenes": len(scenes),
            "removed_scenes": len(removed_runs),
            "compressed_filename": compressed_filename,
            "compression_stats": compression_stats,
            "scenes_timeline": scenes,
            "unique_scenes_details": scenes,
            "removed_scenes_details": removed_runs,
            "error": None
        }
        return render_template("result.html", stats=stats)
    except Exception as e:
        logging.error(f"[upload] error: {e}\n{traceback.format_exc()}")
        return render_template("result.html", stats=make_error_stats(str(e)))

@app.route("/download/<path:filename>")
def download_file(filename):
    full = os.path.join(app.config["OUTPUT_FOLDER"], filename)
    if not os.path.isfile(full): abort(404)
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename, as_attachment=True)

@app.route("/outputs/<path:filename>")
def serve_output(filename):
    full = os.path.join(app.config["OUTPUT_FOLDER"], filename)
    if not os.path.isfile(full): abort(404)
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
