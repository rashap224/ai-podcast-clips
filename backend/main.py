import glob
import json
import pathlib
import pickle
import shutil
import subprocess
import time
import uuid
import boto3
import cv2
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import ffmpegcv
import modal
import numpy as np
from pydantic import BaseModel
import os
from google import genai
from google.genai import types

import pysubs2
from tqdm import tqdm
import whisperx


class ProcessVideoRequest(BaseModel):
    s3_key: str
    # Optional: per-request Gemini model override (e.g., "gemini-2.5-pro", "gemini-2.5-flash")
    model: Optional[str] = None
    # Optional: compute global ASD to bias candidate selection (heavier but more accurate)
    compute_asd: Optional[bool] = True

THIS_DIR = os.path.dirname(__file__)
REQ_FILE = os.path.join(THIS_DIR, "requirements.txt")
LR_ASD_DIR = os.path.join(THIS_DIR, "LR-ASD")

image = (modal.Image.from_registry(
    "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install(["ffmpeg", "libgl1-mesa-glx", "wget", "libcudnn8", "libcudnn8-dev"])
    .pip_install_from_requirements(REQ_FILE)
    .run_commands(["mkdir -p /usr/share/fonts/truetype/custom",
                   "wget -O /usr/share/fonts/truetype/custom/Anton-Regular.ttf https://github.com/google/fonts/raw/main/ofl/anton/Anton-Regular.ttf",
                   "fc-cache -f -v"])
    .add_local_dir(LR_ASD_DIR, "/LR-ASD", copy=True))

app = modal.App("ai-podcast-clips", image=image)

volume = modal.Volume.from_name(
    "ai-podcast-clips-model-cache", create_if_missing=True
)

mount_path = "/root/.cache/torch"

auth_scheme = HTTPBearer()


def create_vertical_video(tracks, scores, pyframes_path, pyavi_path, audio_path, output_path, framerate=25):
    target_width = 1080
    target_height = 1920

    flist = glob.glob(os.path.join(pyframes_path, "*.jpg"))
    flist.sort()

    faces = [[] for _ in range(len(flist))]

    for tidx, track in enumerate(tracks):
        score_array = scores[tidx]
        for fidx, frame in enumerate(track["track"]["frame"].tolist()):
            slice_start = max(fidx - 30, 0)
            slice_end = min(fidx + 30, len(score_array))
            score_slice = score_array[slice_start:slice_end]
            avg_score = float(np.mean(score_slice)
                              if len(score_slice) > 0 else 0)

            faces[frame].append(
                {'track': tidx, 'score': avg_score, 's': track['proc_track']["s"][fidx], 'x': track['proc_track']["x"][fidx], 'y': track['proc_track']["y"][fidx]})

    temp_video_path = os.path.join(pyavi_path, "video_only.mp4")

    vout = None
    for fidx, fname in tqdm(enumerate(flist), total=len(flist), desc="Creating vertical video"):
        img = cv2.imread(fname)
        if img is None:
            continue

        current_faces = faces[fidx]

        max_score_face = max(
            current_faces, key=lambda face: face['score']) if current_faces else None

        if max_score_face and max_score_face['score'] < 0:
            max_score_face = None

        if vout is None:
            vout = ffmpegcv.VideoWriterNV(
                file=temp_video_path,
                codec=None,
                fps=framerate,
                resize=(target_width, target_height)
            )

        if max_score_face:
            mode = "crop"
        else:
            mode = "resize"

        if mode == "resize":
            scale = target_width / img.shape[1]
            resized_height = int(img.shape[0] * scale)
            resized_image = cv2.resize(
                img, (target_width, resized_height), interpolation=cv2.INTER_AREA)

            scale_for_bg = max(
                target_width / img.shape[1], target_height / img.shape[0])
            bg_width = int(img.shape[1] * scale_for_bg)
            bg_heigth = int(img.shape[0] * scale_for_bg)

            blurred_background = cv2.resize(img, (bg_width, bg_heigth))
            blurred_background = cv2.GaussianBlur(
                blurred_background, (121, 121), 0)

            crop_x = (bg_width - target_width) // 2
            crop_y = (bg_heigth - target_height) // 2
            blurred_background = blurred_background[crop_y:crop_y +
                                                    target_height, crop_x:crop_x + target_width]

            center_y = (target_height - resized_height) // 2
            blurred_background[center_y:center_y +
                               resized_height, :] = resized_image

            vout.write(blurred_background)

        elif mode == "crop":
            scale = target_height / img.shape[0]
            resized_image = cv2.resize(
                img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frame_width = resized_image.shape[1]

            center_x = int(
                max_score_face["x"] * scale if max_score_face else frame_width // 2)
            top_x = max(min(center_x - target_width // 2,
                        frame_width - target_width), 0)

            image_cropped = resized_image[0:target_height,
                                          top_x:top_x + target_width]

            vout.write(image_cropped)

    if vout:
        vout.release()

    ffmpeg_command = (f"ffmpeg -y -i {temp_video_path} -i {audio_path} "
                      f"-c:v h264 -preset fast -crf 23 -c:a aac -b:a 128k "
                      f"{output_path}")
    subprocess.run(ffmpeg_command, shell=True, check=True, text=True)


def create_subtitles_with_ffmpeg(transcript_segments: list, clip_start: float, clip_end: float, clip_video_path: str, output_path: str, max_words: int = 5):
    temp_dir = os.path.dirname(output_path)
    subtitle_path = os.path.join(temp_dir, "temp_subtitles.ass")

    clip_segments = [segment for segment in transcript_segments
                     if segment.get("start") is not None
                     and segment.get("end") is not None
                     and segment.get("end") > clip_start
                     and segment.get("start") < clip_end
                     ]

    subtitles = []
    current_words = []
    current_start = None
    current_end = None

    for segment in clip_segments:
        word = segment.get("word", "").strip()
        seg_start = segment.get("start")
        seg_end = segment.get("end")

        if not word or seg_start is None or seg_end is None:
            continue

        start_rel = max(0.0, seg_start - clip_start)
        end_rel = max(0.0, seg_end - clip_start)

        if end_rel <= 0:
            continue

        if not current_words:
            current_start = start_rel
            current_end = end_rel
            current_words = [word]
        elif len(current_words) >= max_words:
            subtitles.append(
                (current_start, current_end, ' '.join(current_words)))
            current_words = [word]
            current_start = start_rel
            current_end = end_rel
        else:
            current_words.append(word)
            current_end = end_rel

    if current_words:
        subtitles.append(
            (current_start, current_end, ' '.join(current_words)))

    subs = pysubs2.SSAFile()

    subs.info["WrapStyle"] = 0
    subs.info["ScaledBorderAndShadow"] = "yes"
    subs.info["PlayResX"] = 1080
    subs.info["PlayResY"] = 1920
    subs.info["ScriptType"] = "v4.00+"

    style_name = "Default"
    new_style = pysubs2.SSAStyle()
    new_style.fontname = "Anton"
    new_style.fontsize = 140
    new_style.primarycolor = pysubs2.Color(255, 255, 255)
    new_style.outline = 2.0
    new_style.shadow = 2.0
    new_style.shadowcolor = pysubs2.Color(0, 0, 0, 128)
    new_style.alignment = 2
    new_style.marginl = 50
    new_style.marginr = 50
    new_style.marginv = 50
    new_style.spacing = 0.0

    subs.styles[style_name] = new_style

    for i, (start, end, text) in enumerate(subtitles):
        start_time = pysubs2.make_time(s=start)
        end_time = pysubs2.make_time(s=end)
        line = pysubs2.SSAEvent(
            start=start_time, end=end_time, text=text, style=style_name)
        subs.events.append(line)

    subs.save(subtitle_path)

    ffmpeg_cmd = (f"ffmpeg -y -i {clip_video_path} -vf \"ass={subtitle_path}\" "
                  f"-c:v h264 -preset fast -crf 23 {output_path}")

    subprocess.run(ffmpeg_cmd, shell=True, check=True)


def process_clip(base_dir: str, original_video_path: str, s3_key: str, start_time: float, end_time: float, clip_index: int, transcript_segments: list):
    clip_name = f"clip_{clip_index}"
    s3_key_dir = os.path.dirname(s3_key)
    output_s3_key = f"{s3_key_dir}/{clip_name}.mp4"
    print(f"Output S3 key: {output_s3_key}")

    clip_dir = base_dir / clip_name
    clip_dir.mkdir(parents=True, exist_ok=True)

    clip_segment_path = clip_dir / f"{clip_name}_segment.mp4"
    vertical_mp4_path = clip_dir / "pyavi" / "video_out_vertical.mp4"
    subtitle_output_path = clip_dir / "pyavi" / "video_with_subtitles.mp4"

    (clip_dir / "pywork").mkdir(exist_ok=True)
    pyframes_path = clip_dir / "pyframes"
    pyavi_path = clip_dir / "pyavi"
    audio_path = clip_dir / "pyavi" / "audio.wav"

    pyframes_path.mkdir(exist_ok=True)
    pyavi_path.mkdir(exist_ok=True)

    duration = end_time - start_time
    cut_command = (f"ffmpeg -i {original_video_path} -ss {start_time} -t {duration} "
                   f"{clip_segment_path}")
    subprocess.run(cut_command, shell=True, check=True,
                   capture_output=True, text=True)

    extract_cmd = f"ffmpeg -i {clip_segment_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
    subprocess.run(extract_cmd, shell=True,
                   check=True, capture_output=True)

    shutil.copy(clip_segment_path, base_dir / f"{clip_name}.mp4")

    columbia_command = (f"python Columbia_test.py --videoName {clip_name} "
                        f"--videoFolder {str(base_dir)} "
                        f"--pretrainModel weight/finetuning_TalkSet.model")

    columbia_start_time = time.time()
    subprocess.run(columbia_command, cwd="/LR-ASD", shell=True)
    columbia_end_time = time.time()
    print(
        f"Columbia script completed in {columbia_end_time - columbia_start_time:.2f} seconds")

    tracks_path = clip_dir / "pywork" / "tracks.pckl"
    scores_path = clip_dir / "pywork" / "scores.pckl"
    if not tracks_path.exists() or not scores_path.exists():
        raise FileNotFoundError("Tracks or scores not found for clip")

    with open(tracks_path, "rb") as f:
        tracks = pickle.load(f)

    with open(scores_path, "rb") as f:
        scores = pickle.load(f)

    cvv_start_time = time.time()
    create_vertical_video(
        tracks, scores, pyframes_path, pyavi_path, audio_path, vertical_mp4_path
    )
    cvv_end_time = time.time()
    print(
        f"Clip {clip_index} vertical video creation time: {cvv_end_time - cvv_start_time:.2f} seconds")

    create_subtitles_with_ffmpeg(transcript_segments, start_time,
                                 end_time, vertical_mp4_path, subtitle_output_path, max_words=5)

    s3_client = boto3.client("s3")
    s3_client.upload_file(
        subtitle_output_path, "clips-podcast", output_s3_key)


@app.cls(gpu="L40S", timeout=900, retries=0, scaledown_window=20, secrets=[modal.Secret.from_name("ai-podcast-clips-secret")], volumes={mount_path: volume})
class AiPodcastClips:
    @modal.enter()
    def load_model(self):
        print("Loading models")

        self.whisperx_model = whisperx.load_model(
            "large-v2", device="cuda", compute_type="float16")

        self.alignment_model, self.metadata = whisperx.load_align_model(
            language_code="en",
            device="cuda"
        )

        print("Transcription models loaded...")

        print("Creating gemini client...")
        gem_api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not gem_api_key:
            raise RuntimeError(
                "Missing GEMINI_API_KEY (or GOOGLE_API_KEY). Configure it via a Modal secret 'ai-podcast-clips-secret' or environment variable."
            )
        self.gemini_client = genai.Client(api_key=gem_api_key) 
        # Default to balanced price/performance; allow override via GEMINI_MODEL
        self.gen_model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        print("Created gemini client...")

    def transcribe_video(self, base_dir: str, video_path: str) -> str:
        audio_path = base_dir / "audio.wav"
        extract_cmd = f"ffmpeg -i {video_path} -vn -acodec pcm_s16le -ar 16000 -ac 1 {audio_path}"
        subprocess.run(extract_cmd, shell=True,
                       check=True, capture_output=True)

        print("Starting transcription with WhisperX...")
        start_time = time.time()

        audio = whisperx.load_audio(str(audio_path))
        result = self.whisperx_model.transcribe(audio, batch_size=16)

        result = whisperx.align(
            result["segments"],
            self.alignment_model,
            self.metadata,
            audio,
            device="cuda",
            return_char_alignments=False
        )

        duration = time.time() - start_time
        print("Transcription and alignment took " + str(duration) + " seconds")

        segments = []

        if "word_segments" in result:
            for word_segment in result["word_segments"]:
                segments.append({
                    "start": word_segment["start"],
                    "end": word_segment["end"],
                    "word": word_segment["word"],
                })

        return json.dumps(segments)

    # --- Helpers to structure transcript into sentences and candidates ---
    def words_to_sentences(self, words: list, max_gap: float = 0.6, max_words: int = 30):
        sentences = []
        cur_words = []
        cur_start = None
        prev_end = None

        def flush():
            if cur_words and cur_start is not None and prev_end is not None:
                sentences.append({
                    "start": cur_start,
                    "end": prev_end,
                    "text": " ".join(cur_words)
                })

        for w in words:
            wstart = w.get("start")
            wend = w.get("end")
            wtext = (w.get("word") or "").strip()
            if wstart is None or wend is None or not wtext:
                continue

            # Start a new sentence on long pause or length cap
            if prev_end is not None and ((wstart - prev_end) > max_gap or len(cur_words) >= max_words):
                flush()
                cur_words = []
                cur_start = None

            if not cur_words:
                cur_start = wstart
            cur_words.append(wtext)
            prev_end = wend

        flush()
        return sentences

    def propose_candidates(self, sentences: list, min_len: float = 30.0, max_len: float = 60.0, stride: int = 1, asd_scores: Optional[list] = None, asd_fps: Optional[float] = None):
        def q_score(text: str) -> int:
            toks = text.lower().split()
            qs = {"who", "what", "why", "how", "when", "where", "did", "do", "does", "can", "should", "would", "is", "are", "could"}
            return sum(1 for t in toks if t in qs)

        def mean_asd(start_s: float, end_s: float) -> float:
            if not asd_scores or not asd_fps or end_s <= start_s:
                return 0.0
            s_idx = max(0, int(start_s * asd_fps))
            e_idx = min(len(asd_scores), int(end_s * asd_fps))
            if e_idx <= s_idx:
                return 0.0
            window = asd_scores[s_idx:e_idx]
            if not window:
                return 0.0
            return float(np.mean(window))

        cands = []
        n = len(sentences)
        for i in range(0, n, stride):
            cur_start = sentences[i]["start"]
            cur_end = sentences[i]["end"]
            cur_text = [sentences[i]["text"]]
            score = q_score(sentences[i]["text"])
            j = i + 1
            while j < n and (cur_end - cur_start) < max_len:
                cur_end = sentences[j]["end"]
                cur_text.append(sentences[j]["text"])
                score += q_score(sentences[j]["text"])
                if (cur_end - cur_start) >= min_len:
                    text_concat = " ".join(cur_text)
                    # words per second as a weak proxy for energy
                    dur = max(1e-6, (cur_end - cur_start))
                    wps = len(text_concat.split()) / dur
                    asd_mean = mean_asd(cur_start, cur_end)
                    # Composite score: question density + pace + ASD bias
                    composite = score + 0.1 * wps + 1.0 * asd_mean
                    cands.append({
                        "start": cur_start,
                        "end": cur_end,
                        "text": text_concat[:2000],
                        "score": composite,
                        "asd": asd_mean
                    })
                j += 1

        cands.sort(key=lambda c: c["score"], reverse=True)
        return cands[:20]

    def identify_moments(self, transcript: list, model_name: Optional[str] = None, asd_scores: Optional[list] = None, asd_fps: Optional[float] = None):
        # transcript: list of {start, end, word}
        try:
            sentences = self.words_to_sentences(transcript, max_gap=0.6, max_words=30)
        except Exception as e:
            print(f"Sentenceization failed: {e}")
            sentences = []
        if not sentences:
            return "[]"

        candidates = self.propose_candidates(
            sentences,
            min_len=30.0,
            max_len=60.0,
            stride=1,
            asd_scores=asd_scores,
            asd_fps=asd_fps,
        )
        if not candidates:
            return "[]"

        # Few-shot examples (small) to steer toward Q->A/story arc and non-overlap
        few_shot = """
Examples:
Candidates: [{"start": 0, "end": 35, "text": "Why do habits stick?"},
 {"start": 35, "end": 68, "text": "They wire in via repetition..."},
 {"start": 70, "end": 95, "text": "Thanks for listening"}]
Good output: [{"start": 0, "end": 68}]
Bad output (avoid greeting-only or overlapping): []
"""

        prompt_parts = [
            "Select non-overlapping, viral podcast clip windows (30–60s) from the candidates. ",
            "Prefer question→answer or a concise story arc. Use candidates' timestamps exactly. ",
            'Avoid greetings/thanks-only. Return JSON array: [{"start": number, "end": number}].\n',
            few_shot,
            "Candidates:\n",
        ]
        prompt = "".join(prompt_parts)

        compact = [{"start": c["start"], "end": c["end"], "text": c["text"]} for c in candidates]

        model_to_use = model_name or self.gen_model_name
        print(f"Gemini model: {model_to_use}")
        response = self.gemini_client.models.generate_content(
            model=model_to_use,
            contents=prompt + json.dumps(compact),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "start": types.Schema(type=types.Type.NUMBER),
                            "end": types.Schema(type=types.Type.NUMBER),
                        },
                        required=["start", "end"],
                    ),
                ),
                temperature=0.2,
                top_p=0.95,
                top_k=32,
            ),
        )
        # Log usage if SDK provides it (input/output tokens, etc.)
        usage = getattr(response, "usage_metadata", None)
        if usage:
            try:
                print(f"Gemini usage: {usage}")
            except Exception:
                pass
        print(f"Identified moments response: {response.text}")
        return response.text

    # --- Active Speaker Detection (global) ---
    def compute_global_asd(self, base_dir: pathlib.Path, video_path: pathlib.Path) -> tuple[Optional[list], Optional[float]]:
        """Run Columbia ASD on the full input video to estimate active speaker per frame.
        Returns (frame_scores, fps). If it fails, returns (None, None).
        """
        try:
            # Reuse the ASD pipeline by pointing to the base video name "input"
            columbia_command = (f"python Columbia_test.py --videoName input "
                                f"--videoFolder {str(base_dir)} "
                                f"--pretrainModel weight/finetuning_TalkSet.model")
            start = time.time()
            subprocess.run(columbia_command, cwd="/LR-ASD", shell=True, check=True)
            print(f"Global ASD completed in {time.time()-start:.2f}s")

            # Outputs are stored under {base_dir}/input/pywork
            work_dir = base_dir / "input" / "pywork"
            tracks_path = work_dir / "tracks.pckl"
            scores_path = work_dir / "scores.pckl"
            if not tracks_path.exists() or not scores_path.exists():
                print("Global ASD: tracks/scores not found")
                return None, None

            with open(tracks_path, "rb") as f:
                tracks = pickle.load(f)
            with open(scores_path, "rb") as f:
                scores = pickle.load(f)

            # Aggregate per-frame max score across tracks
            # Determine number of frames by scanning track frame indices
            max_frame = 0
            for track in tracks:
                frames = track["track"]["frame"].tolist()
                if frames:
                    max_frame = max(max_frame, int(max(frames)))
            total_frames = max_frame + 1
            frame_scores = np.zeros(total_frames, dtype=np.float32)

            for tidx, track in enumerate(tracks):
                score_array = scores[tidx]
                for fidx, frame in enumerate(track["track"]["frame" ].tolist()):
                    if 0 <= frame < total_frames and 0 <= fidx < len(score_array):
                        frame_scores[frame] = max(frame_scores[frame], float(score_array[fidx]))

            # Assume 25 fps for ASD frames (Columbia pipeline default in this repo)
            fps = 25.0
            return frame_scores.tolist(), fps
        except Exception as e:
            print(f"Global ASD failed: {e}")
            return None, None

    # --- Post process to sentence boundaries and enforce 30–60s non-overlap ---
    def enforce_clip_bounds(self, clips: list, sentences: list, min_len: float = 30.0, max_len: float = 60.0):
        if not clips:
            return []
        # Helper to find sentence index by time
        starts = [s["start"] for s in sentences]
        ends = [s["end"] for s in sentences]

        def nearest_start(t: float) -> float:
            # choose sentence start at/after t if possible, else closest prior
            for s in sentences:
                if s["start"] >= t:
                    return s["start"]
            return sentences[-1]["start"]

        def nearest_end(t: float) -> float:
            # choose sentence end at/after t
            for s in sentences:
                if s["end"] >= t:
                    return s["end"]
            return sentences[-1]["end"]

        adjusted = []
        for c in clips:
            s = float(c.get("start", 0.0))
            e = float(c.get("end", 0.0))
            if e <= s:
                continue
            ns = nearest_start(s)
            ne = nearest_end(e)
            # Expand to min_len if needed
            if (ne - ns) < min_len:
                # expand end forward along sentence ends
                for sent in sentences:
                    if sent["end"] > ne:
                        ne = sent["end"]
                        if (ne - ns) >= min_len:
                            break
            # Trim to max_len if needed
            if (ne - ns) > max_len:
                # find last sentence end within max_len
                acc_end = ns
                for sent in sentences:
                    if sent["end"] <= ns:
                        continue
                    next_end = sent["end"]
                    if (next_end - ns) <= max_len:
                        acc_end = next_end
                    else:
                        break
                ne = max(acc_end, min(ns + min_len, ne))
            adjusted.append({"start": ns, "end": ne})

        # Enforce non-overlap by trimming or dropping
        adjusted.sort(key=lambda x: x["start"]) 
        non_overlap = []
        last_end = -1e9
        for c in adjusted:
            if c["start"] >= last_end:
                non_overlap.append(c)
                last_end = c["end"]
            else:
                # try to trim start to last_end via next sentence start
                ns = c["start"]
                for sent in sentences:
                    if sent["start"] >= last_end:
                        ns = sent["start"]
                        break
                if (c["end"] - ns) >= min_len:
                    non_overlap.append({"start": ns, "end": c["end"]})
                    last_end = c["end"]
                # else drop
        return non_overlap

    @modal.fastapi_endpoint(method="POST")
    def process_video(self, request: ProcessVideoRequest, token: HTTPAuthorizationCredentials = Depends(auth_scheme)):
        s3_key = request.s3_key

        expected_token = os.environ.get("AUTH_TOKEN")
        if expected_token is None:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                                detail="Server not configured: AUTH_TOKEN is missing. Add it to your Modal secret or environment.")
        if token.credentials != expected_token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                detail="Incorrect bearer token", headers={"WWW-Authenticate": "Bearer"})

        run_id = str(uuid.uuid4())
        base_dir = pathlib.Path("/tmp") / run_id
        base_dir.mkdir(parents=True, exist_ok=True)

        # Download video file
        video_path = base_dir / "input.mp4"
        s3_client = boto3.client("s3")
        s3_client.download_file("clips-podcast", s3_key, str(video_path))

        # 1. Transcription
        transcript_segments_json = self.transcribe_video(base_dir, video_path)
        transcript_segments = json.loads(transcript_segments_json)

        # Optional: compute global ASD to bias candidate scoring
        asd_scores = None
        asd_fps = None
        if request.compute_asd:
            print("Computing global ASD for candidate biasing...")
            asd_scores, asd_fps = self.compute_global_asd(base_dir, video_path)

        # 2. Identify moments for clips
        print("Identifying clip moments")
        # Allow per-request model override
        selected_model = request.model or os.environ.get("GEMINI_MODEL", self.gen_model_name)
        identified_moments_raw = self.identify_moments(transcript_segments, model_name=selected_model, asd_scores=asd_scores, asd_fps=asd_fps)

        # JSON mode should already return parseable JSON
        try:
            clip_moments = json.loads(identified_moments_raw)
            if not isinstance(clip_moments, list):
                print("Identify moments output not a list; using empty list")
                clip_moments = []
        except Exception as e:
            print(f"Identify moments parse error: {e}")
            clip_moments = []

        # 2.1 Post-process to sentence boundaries and enforce 30–60s non-overlap
        sentences_for_bounds = self.words_to_sentences(transcript_segments, max_gap=0.6, max_words=30)
        clip_moments = self.enforce_clip_bounds(clip_moments, sentences_for_bounds, min_len=30.0, max_len=60.0)
        print(clip_moments)

        # 3. Process clips
        for index, moment in enumerate(clip_moments[:5]):
            if "start" in moment and "end" in moment:
                print("Processing clip" + str(index) + " from " +
                      str(moment["start"]) + " to " + str(moment["end"]))
                process_clip(base_dir, video_path, s3_key,
                             moment["start"], moment["end"], index, transcript_segments)

        if base_dir.exists():
            print(f"Cleaning up temp dir after {base_dir}")
            shutil.rmtree(base_dir, ignore_errors=True)


@app.local_entrypoint()
def main():
    import requests

    ai_podcast_clips = AiPodcastClips()

    url = ai_podcast_clips.process_video.web_url

    payload = {
        "s3_key": "test1/health5min.mp4"
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer 123123"
    }

    response = requests.post(url, json=payload,
                             headers=headers)
    response.raise_for_status()
    result = response.json()
    print(result)