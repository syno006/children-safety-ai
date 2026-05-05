"""Unified DangerousObjectDetection workspace views."""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, render
from django.views.decorators.http import require_POST

from .models import Source
from .registry import pipeline_manifest, pipeline_map, run_pipeline
from .utils import decode_base64_image


# Codecs that OpenCV server builds cannot decode reliably (no HW accel).
_TRANSCODE_CODECS = {'av1', 'hevc', 'vp9', 'vp8', 'av01'}


def _detect_video_codec(video_path: str) -> str:
	"""Return the lowercase codec name via ffprobe, or empty string on failure."""
	import subprocess, shutil
	if not shutil.which('ffprobe'):
		return ''
	try:
		r = subprocess.run(
			['ffprobe', '-v', 'error', '-select_streams', 'v:0',
			 '-show_entries', 'stream=codec_name',
			 '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
			capture_output=True, text=True, timeout=15,
		)
		return r.stdout.strip().lower()
	except Exception:
		return ''


def _transcode_to_h264(video_path: str) -> str:
	"""Re-encode a video to H.264/mp4 when the codec is not OpenCV-friendly.

	Detects the codec with ffprobe first — only transcodes when necessary.
	Returns the path to use (transcoded or original).
	"""
	import subprocess, shutil

	# Return a cached result if we already transcoded this exact file.
	transcoded = video_path.rsplit('.', 1)[0] + '_h264.mp4'
	if os.path.exists(transcoded):
		return transcoded

	codec = _detect_video_codec(video_path)
	if codec and codec not in _TRANSCODE_CODECS:
		return video_path  # Known-good codec — no transcode needed.

	# Either ffprobe said it's a problem codec, or ffprobe is unavailable
	# (codec == '') — transcode unconditionally in that case too, because
	# AV1 cap.read() can return ok=True on frame 0 then silently fail later.
	if not shutil.which('ffmpeg'):
		return video_path  # Can't transcode — return original and surface the error.

	try:
		result = subprocess.run(
			[
				'ffmpeg', '-y',
				'-i', video_path,
				'-c:v', 'libx264',
				'-preset', 'fast',
				'-crf', '23',
				'-c:a', 'aac',
				'-movflags', '+faststart',
				transcoded,
			],
			stdout=subprocess.DEVNULL,
			stderr=subprocess.DEVNULL,
			timeout=300,
		)
		if result.returncode == 0 and os.path.exists(transcoded):
			return transcoded
	except Exception:
		pass

	return video_path  # Transcode failed — return original.


def _toast_events_from_results(results: dict) -> list[dict]:
	"""Create a lightweight list of events for the UI (for toast notifications)."""
	events: list[dict] = []
	for pipeline_id, result in (results or {}).items():
		try:
			score = float(result.get('score', 0) or 0)
		except Exception:
			score = 0.0
		threshold = float(result.get('threshold', 0) or 0)
		if score >= threshold and score > 0:
			events.append({
				'type': 'detection',
				'pipeline': pipeline_id,
				'label': result.get('label', 'Detected'),
				'score': score,
				'risk_level': result.get('risk_level', 'HIGH'),
			})
	return events


def _best_annotated_artifact(results: dict) -> str | None:
	"""Pick the best annotated image artifact from results.

	Priority:
	1) __combined__ artifact (multi-model overlay)
	2) first pipeline artifact

	Returns base64 jpeg/png content (without data: prefix).
	"""
	if not results:
		return None
	combined = results.get('__combined__')
	if isinstance(combined, dict) and combined.get('artifact'):
		return combined.get('artifact')
	for _pid, res in results.items():
		if _pid == '__combined__':
			continue
		if isinstance(res, dict) and res.get('artifact'):
			return res.get('artifact')
	return None


def _draw_combined_overlay(frame_rgb, detections_by_pipeline: dict) -> str | None:
	"""Draw detections from ALL pipelines onto one frame and return base64-encoded image.

	Each pipeline gets a distinct color so boxes are visually distinguishable.
	frame_rgb must be an RGB numpy array.
	"""
	if frame_rgb is None:
		return None

	try:
		import cv2
		import numpy as np
	except Exception:
		return None

	# Defensive copy — never mutate the caller's frame.
	img = np.array(frame_rgb, copy=True)
	if img.ndim != 3 or img.shape[2] != 3:
		return None

	img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

	# One distinct color per pipeline (BGR).
	colors = [
		(0, 0, 255),    # red     — pipeline 0
		(0, 200, 80),   # green   — pipeline 1
		(255, 140, 0),  # blue    — pipeline 2
		(200, 0, 200),  # magenta — pipeline 3
	]

	for idx, (pipeline_id, dets) in enumerate(detections_by_pipeline.items()):
		color = colors[idx % len(colors)]
		for det in (dets or []):
			bbox = det.get('bbox') or []
			if len(bbox) != 4:
				continue
			try:
				x1, y1, x2, y2 = [int(max(0, float(v))) for v in bbox]
			except (TypeError, ValueError):
				continue
			label = det.get('label', pipeline_id)
			conf = float(det.get('confidence', 0))
			cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
			text = f"{label} {conf:.0%}"
			(tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
			cv2.rectangle(img_bgr, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
			cv2.putText(img_bgr, text, (x1 + 3, y1 - 4),
						cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

	from .utils import numpy_to_b64
	return numpy_to_b64(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def index(request):
	if request.user.is_authenticated:
		sources = list(Source.objects.filter(owner=request.user))
	else:
		sources = list(Source.objects.all())
	pipelines = pipeline_manifest()

	for source in sources:
		_ensure_model_config(source, pipelines)
		source.media_url = _media_url_from_path(source.last_media_path)
		source.layout_json = json.dumps(source.layout or {})
		source.config_json = json.dumps(source.model_config or {})
		source.camera_json = json.dumps(source.camera_config or {})

	stats = {
		'sources': len(sources),
		'pipelines': len(pipelines),
		'enabled': sum(
			1 for source in sources for cfg in (source.model_config or {}).values() if cfg.get('enabled')
		),
	}

	return render(request, 'dangerous_detector/index.html', {
		'sources': sources,
		'pipelines': pipelines,
		'stats': stats,
	})


@require_POST
def create_source(request):
	payload = json.loads(request.body or '{}')
	name = payload.get('name') or f"Source {Source.objects.count() + 1}"
	source_type = payload.get('source_type', 'image')
	camera_config = payload.get('camera_config') or {}

	source = Source.objects.create(
		name=name,
		owner=request.user if request.user.is_authenticated else None,
		source_type=source_type,
		model_config=_default_model_config(pipeline_manifest()),
		camera_config=camera_config,
	)
	return JsonResponse({
		'id': source.id,
		'name': source.name,
		'source_type': source.source_type,
	})


@require_POST
def update_layout(request, source_id: int):
	source = get_object_or_404(Source, pk=source_id)
	payload = json.loads(request.body or '{}')
	source.layout = payload.get('layout', {})
	source.save(update_fields=['layout', 'updated_at'])
	return JsonResponse({'status': 'ok'})


@require_POST
def update_models(request, source_id: int):
	source = get_object_or_404(Source, pk=source_id)
	payload = json.loads(request.body or '{}')
	model_config = payload.get('model_config', {})
	source.model_config = model_config
	source.save(update_fields=['model_config', 'updated_at'])
	return JsonResponse({'status': 'ok'})


@require_POST
def update_camera_config(request, source_id: int):
	source = get_object_or_404(Source, pk=source_id)
	payload = json.loads(request.body or '{}')
	camera_config = payload.get('camera_config', {})
	source.camera_config = camera_config
	source.save(update_fields=['camera_config', 'updated_at'])
	return JsonResponse({'status': 'ok'})


@require_POST
def run_source(request, source_id: int):
	source = get_object_or_404(Source, pk=source_id)
	media_path = source.last_media_path
	media_name = source.last_media_name

	if request.FILES:
		file_key = 'media'
		uploaded = request.FILES.get(file_key)
		if uploaded:
			ext = Path(uploaded.name).suffix.lower()
			unique_name = f"{uuid.uuid4().hex}{ext}"
			upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads', 'dangerous')
			os.makedirs(upload_dir, exist_ok=True)
			fs = FileSystemStorage(location=upload_dir)
			filename = fs.save(unique_name, uploaded)
			media_path = os.path.join(upload_dir, filename)
			media_name = uploaded.name

	if not media_path:
		return JsonResponse({'error': 'No media uploaded yet.'}, status=400)

	# Video files cannot be processed as a single image — hand off to the
	# video-live worker which samples every frame and streams results back.
	VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
	if Path(media_path).suffix.lower() in VIDEO_EXTS or source.source_type == 'video':
		# Delegate to run_video_live by forwarding the request there.
		# We re-use the same request object; run_video_live reads request.FILES
		# for a new upload or falls through to error. Since the file is already
		# saved, we inject the path into the job directly.
		import cv2, threading, time as _time

		global _VIDEO_JOBS  # type: ignore
		try:
			_VIDEO_JOBS
		except NameError:
			_VIDEO_JOBS = {}

		upload_dir = os.path.dirname(media_path)
		job_id = uuid.uuid4().hex
		_VIDEO_JOBS[job_id] = {'done': False, 'progress': 0, 'frame': None, 'events': [], 'media_url': ''}

		def _worker():
			from .utils import numpy_to_b64
			from .registry import annotate_frame_bgr_with_yolo

			work_path = _transcode_to_h264(media_path)
			cap = cv2.VideoCapture(work_path)
			if not cap.isOpened():
				_VIDEO_JOBS[job_id]['done'] = True
				_VIDEO_JOBS[job_id]['events'].append({'type': 'error', 'message': 'Unable to open video.'})
				return

			frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
			fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
			if fps <= 0 or fps > 120:
				fps = 25.0

			width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
			height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
			if width <= 0 or height <= 0:
				width, height = 1280, 720

			annotated_name = f"annotated_{job_id}.mp4"
			annotated_path = os.path.join(upload_dir, annotated_name)
			fourcc = cv2.VideoWriter_fourcc(*'mp4v')
			writer = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))

			preview_step = max(1, int(fps // 4))
			idx = 0
			try:
				while True:
					ok, frame_bgr = cap.read()
					if not ok or frame_bgr is None:
						break
					idx += 1
					if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
						frame_bgr = cv2.resize(frame_bgr, (width, height))

					# Annotate like detect_final.py (in-place BGR drawing) with BOTH models.
					frame_for_models = frame_bgr
					detections_by_pipeline: dict[str, list[dict]] = {}
					max_score = 0.0
					for pipeline_id, cfg in (source.model_config or {}).items():
						# Respect enabled/threshold per source.
						if cfg is not None and isinstance(cfg, dict) and not cfg.get('enabled', True):
							continue
						threshold = float((cfg or {}).get('threshold', 50))
						# Only our two pipelines are exposed, so we can map ids → model paths.
						if pipeline_id == 'fire_detection':
							model_path = str(settings.BASE_DIR / 'models' / 'FireDetec' / 'bestF.pt')
						elif pipeline_id == 'dangerous_object_i':
							model_path = str(settings.BASE_DIR / 'models' / 'dangerousOb' / 'bestI.pt')
						else:
							continue
						frame_for_models, dets, max_conf = annotate_frame_bgr_with_yolo(
							model_path,
							frame_for_models,
							threshold=threshold,
						)
						if dets:
							detections_by_pipeline[pipeline_id] = dets
						max_score = max(max_score, float(max_conf * 100.0))

					# Write every frame to output at full FPS.
					writer.write(frame_for_models)

					# Build events in the same shape as before for the UI.
					events = []
					for pid, dets in detections_by_pipeline.items():
						try:
							threshold = float((source.model_config or {}).get(pid, {}).get('threshold', 50))
						except Exception:
							threshold = 50.0
						score = max([float(d.get('confidence', 0)) for d in dets] + [0.0]) * 100.0
						if score >= threshold and score > 0:
							events.append({
								'type': 'detection',
								'pipeline': pid,
								'label': (dets[0].get('label') if dets else 'Detected'),
								'score': score,
								'risk_level': 'HIGH',
							})

					_VIDEO_JOBS[job_id]['progress'] = int((idx / max(1, frame_count)) * 100)
					_VIDEO_JOBS[job_id]['events'].extend(events)
					if idx % preview_step == 0:
						# Preview uses base64 JPEG/PNG, but the output mp4 is written from BGR frames.
						preview_rgb = cv2.cvtColor(frame_for_models, cv2.COLOR_BGR2RGB)
						preview_b64 = numpy_to_b64(preview_rgb)
						if preview_b64:
							_VIDEO_JOBS[job_id]['frame'] = preview_b64
			finally:
				cap.release()
				try:
					writer.release()
				except Exception:
					pass

			_VIDEO_JOBS[job_id]['done'] = True
			_VIDEO_JOBS[job_id]['media_url'] = _media_url_from_path(annotated_path)
			source.last_media_path = media_path
			source.last_media_name = media_name or source.last_media_name
			source.save(update_fields=['last_media_path', 'last_media_name', 'updated_at'])

		threading.Thread(target=_worker, daemon=True).start()
		return JsonResponse({'job_id': job_id, 'done': False, 'progress': 0, 'is_video': True})

	# Image file — run synchronously.
	results = _run_models_for_source(source, media_path=media_path)

	source.last_media_path = media_path
	source.last_media_name = media_name or source.last_media_name
	source.last_results = results
	source.save(update_fields=['last_media_path', 'last_media_name', 'last_results', 'updated_at'])

	media_url = _media_url_from_path(media_path)
	annotated = _best_annotated_artifact(results)
	combined = results.pop('__combined__', None)
	events = _toast_events_from_results(results)
	return JsonResponse({
		'results': results,
		'combined': combined,
		'annotated': annotated,
		'events': events,
		'media_url': media_url,
		'media_name': media_name,
	})


@require_POST
def run_frame(request, source_id: int):
	source = get_object_or_404(Source, pk=source_id)
	payload = json.loads(request.body or '{}')
	frame_data = payload.get('image')
	if not frame_data:
		return JsonResponse({'error': 'No frame provided.'}, status=400)

	frame = decode_base64_image(frame_data)
	if frame is None:
		return JsonResponse({'error': 'Failed to decode frame image.'}, status=400)

	results = _run_models_for_source(source, frame=frame)
	source.last_results = results
	source.save(update_fields=['last_results', 'updated_at'])
	annotated = _best_annotated_artifact(results)
	combined = results.pop('__combined__', None)
	events = _toast_events_from_results(results)
	return JsonResponse({'results': results, 'combined': combined, 'annotated': annotated, 'events': events})


@require_POST
def run_stream(request, source_id: int):
	source = get_object_or_404(Source, pk=source_id)
	stream_url = (source.camera_config or {}).get('stream_url')
	if not stream_url:
		return JsonResponse({'error': 'No stream URL configured.'}, status=400)

	import cv2

	# Use FFMPEG backend with TCP transport for RTSP streams — avoids UDP
	# packet loss that causes cap.read() to silently return black frames.
	cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
	if not cap.isOpened():
		# Fallback: try default backend (works for HTTP streams / local devices).
		cap = cv2.VideoCapture(stream_url)
	if not cap.isOpened():
		return JsonResponse({'error': 'Unable to open stream.'}, status=400)

	# Attempt up to 5 reads — first frame from an RTSP stream is often corrupt.
	frame = None
	for _ in range(5):
		ok, frame = cap.read()
		if ok and frame is not None:
			break
	cap.release()
	if frame is None:
		return JsonResponse({'error': 'Unable to read frame from stream.'}, status=400)

	frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	results = _run_models_for_source(source, frame=frame_rgb)
	source.last_results = results
	source.save(update_fields=['last_results', 'updated_at'])
	annotated = _best_annotated_artifact(results)
	combined = results.pop('__combined__', None)
	events = _toast_events_from_results(results)
	return JsonResponse({'results': results, 'combined': combined, 'annotated': annotated, 'events': events})


@require_POST
def run_video_live(request, source_id: int):
	"""Process an uploaded video and return periodic annotated frames (polling).

	UI sends the file in the first request and then keeps polling with job_id.
	This avoids needing WebSockets while still giving a 'live' experience.
	"""
	source = get_object_or_404(Source, pk=source_id)
	import cv2
	import threading
	import time

	# job state is kept in memory (dev friendly). For production use, store in DB/redis.
	global _VIDEO_JOBS  # type: ignore
	try:
		_VIDEO_JOBS
	except NameError:
		_VIDEO_JOBS = {}

	job_id = request.POST.get('job_id') or ''
	if job_id and job_id in _VIDEO_JOBS:
		job = _VIDEO_JOBS[job_id]
		# Return latest frame if available.
		payload = {
			'job_id': job_id,
			'done': job.get('done', False),
			'progress': job.get('progress', 0),
			'frame': job.get('frame'),
			'media_url': job.get('media_url', ''),
			'events': job.get('events', []),
		}
		# clear events after read
		job['events'] = []
		return JsonResponse(payload)

	uploaded = request.FILES.get('media')
	if not uploaded:
		return JsonResponse({'error': 'No video uploaded.'}, status=400)

	# Save uploaded video
	ext = Path(uploaded.name).suffix.lower()
	unique_name = f"{uuid.uuid4().hex}{ext}"
	upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads', 'dangerous')
	os.makedirs(upload_dir, exist_ok=True)
	fs = FileSystemStorage(location=upload_dir)
	filename = fs.save(unique_name, uploaded)
	video_path = os.path.join(upload_dir, filename)

	job_id = uuid.uuid4().hex
	_VIDEO_JOBS[job_id] = {'done': False, 'progress': 0, 'frame': None, 'events': [], 'media_url': ''}

	def _worker():
		from .utils import numpy_to_b64
		from .registry import annotate_frame_bgr_with_yolo

		work_path = _transcode_to_h264(video_path)
		cap = cv2.VideoCapture(work_path)
		if not cap.isOpened():
			_VIDEO_JOBS[job_id]['done'] = True
			_VIDEO_JOBS[job_id]['events'].append({'type': 'error', 'message': 'Unable to open video.'})
			return

		frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
		fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
		if fps <= 0 or fps > 120:
			fps = 25.0

		width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
		height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
		if width <= 0 or height <= 0:
			width, height = 1280, 720

		# Output video is written at the same fps as input so playback speed
		# matches the original. Every frame is annotated and written.
		annotated_name = f"annotated_{job_id}.mp4"
		annotated_path = os.path.join(upload_dir, annotated_name)
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		writer = cv2.VideoWriter(annotated_path, fourcc, fps, (width, height))

		# How often to push a preview frame to the polling client.
		# ~4 previews/sec is enough for a smooth UI without overwhelming it.
		preview_step = max(1, int(fps // 4))

		idx = 0
		try:
			while True:
				ok, frame_bgr = cap.read()
				if not ok or frame_bgr is None:
					break
				idx += 1

				# Resize to declared dimensions if the frame differs (some codecs lie).
				if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
					frame_bgr = cv2.resize(frame_bgr, (width, height))

				# Annotate like detect_final.py (in-place BGR drawing) with BOTH models.
				frame_for_models = frame_bgr
				detections_by_pipeline: dict[str, list[dict]] = {}
				for pipeline_id, cfg in (source.model_config or {}).items():
					if cfg is not None and isinstance(cfg, dict) and not cfg.get('enabled', True):
						continue
					threshold = float((cfg or {}).get('threshold', 50))
					if pipeline_id == 'fire_detection':
						model_path = str(settings.BASE_DIR / 'models' / 'FireDetec' / 'bestF.pt')
					elif pipeline_id == 'dangerous_object_i':
						model_path = str(settings.BASE_DIR / 'models' / 'dangerousOb' / 'bestI.pt')
					else:
						continue
					frame_for_models, dets, _max_conf = annotate_frame_bgr_with_yolo(
						model_path,
						frame_for_models,
						threshold=threshold,
					)
					if dets:
						detections_by_pipeline[pipeline_id] = dets

				# Always write the frame (full FPS output).
				writer.write(frame_for_models)

				# Events for UI.
				events = []
				for pid, dets in detections_by_pipeline.items():
					try:
						threshold = float((source.model_config or {}).get(pid, {}).get('threshold', 50))
					except Exception:
						threshold = 50.0
					score = max([float(d.get('confidence', 0)) for d in dets] + [0.0]) * 100.0
					if score >= threshold and score > 0:
						events.append({
							'type': 'detection',
							'pipeline': pid,
							'label': (dets[0].get('label') if dets else 'Detected'),
							'score': score,
							'risk_level': 'HIGH',
						})

				progress = int((idx / max(1, frame_count)) * 100)
				_VIDEO_JOBS[job_id]['progress'] = progress
				_VIDEO_JOBS[job_id]['events'].extend(events)

				# Push a preview frame to the polling client periodically.
				if idx % preview_step == 0:
					preview_rgb = cv2.cvtColor(frame_for_models, cv2.COLOR_BGR2RGB)
					preview_b64 = numpy_to_b64(preview_rgb)
					if preview_b64:
						_VIDEO_JOBS[job_id]['frame'] = preview_b64
		finally:
			cap.release()
			try:
				writer.release()
			except Exception:
				pass

		_VIDEO_JOBS[job_id]['done'] = True
		_VIDEO_JOBS[job_id]['media_url'] = _media_url_from_path(annotated_path)
		source.last_media_path = video_path
		source.last_media_name = uploaded.name
		source.save(update_fields=['last_media_path', 'last_media_name', 'updated_at'])

	threading.Thread(target=_worker, daemon=True).start()
	# Return immediately so the UI can start polling.
	return JsonResponse({'job_id': job_id, 'done': False, 'progress': 0})


@require_POST
def delete_source(request, source_id: int):
	source = get_object_or_404(Source, pk=source_id)
	source.delete()
	return JsonResponse({'status': 'deleted'})


def _run_models_for_source(source: Source, media_path: str | None = None, frame=None) -> dict:
	config = source.model_config or {}
	pipelines = pipeline_map()
	results = {}
	detections_by_pipeline: dict[str, list[dict]] = {}

	# Determine the effective source type for this run.
	# When running a single frame (camera/stream/video frame), the pipeline should
	# treat it as a camera-like input, not as the persisted source.type.
	if frame is not None:
		effective_source_type = 'camera'
	elif media_path is not None:
		suffix = Path(str(media_path)).suffix.lower()
		if suffix in {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}:
			effective_source_type = 'video'
		else:
			effective_source_type = 'image'
	else:
		effective_source_type = source.source_type

	for pipeline_id, pipeline in pipelines.items():
		cfg = config.get(pipeline_id, {})
		if not cfg.get('enabled', True):
			continue
		threshold = float(cfg.get('threshold', pipeline.default_threshold))
		result = run_pipeline(
			pipeline_id,
			effective_source_type,
			media_path=media_path,
			frame=frame,
			threshold=threshold,
		)
		if result.get('unavailable'):
			continue
		results[pipeline_id] = result
		if result.get('detections'):
			detections_by_pipeline[pipeline_id] = result.get('detections')

	# Always draw ALL detections from ALL pipelines onto a single combined frame.
	# This is the image shown to the user — it must contain boxes from every model.
	#
	# For image/video file uploads, frame is None so we reload the image from disk.
	# For camera/stream, frame is already an RGB numpy array.
	combined_artifact = None

	if detections_by_pipeline:
		# Get the base image to draw on.
		base_frame = frame
		if base_frame is None and media_path:
			from .utils import load_image_from_path
			base_frame = load_image_from_path(media_path)

		if base_frame is not None:
			combined_artifact = _draw_combined_overlay(base_frame, detections_by_pipeline)

	# If no detections at all (models ran but found nothing), still return the
	# plain image as artifact so the UI has something to display.
	if not combined_artifact:
		for _pid, res in (results or {}).items():
			if res.get('artifact'):
				combined_artifact = res.get('artifact')
				break

	if combined_artifact:
		results['__combined__'] = {
			'artifact': combined_artifact,
			'detections': sum(len(v) for v in detections_by_pipeline.values()),
		}

	return results


def _default_model_config(pipelines: list[dict]) -> dict:
	return {
		pipeline['id']: {
			'enabled': True,
			'threshold': pipeline['default_threshold'],
		}
		for pipeline in pipelines
	}


def _ensure_model_config(source: Source, pipelines: list[dict]) -> None:
	changed = False
	current = source.model_config or {}
	for pipeline in pipelines:
		if pipeline['id'] not in current:
			current[pipeline['id']] = {
				'enabled': True,
				'threshold': pipeline['default_threshold'],
			}
			changed = True
	if changed:
		source.model_config = current
		source.save(update_fields=['model_config', 'updated_at'])


def _media_url_from_path(path: str) -> str:
	if not path:
		return ''
	relative = os.path.relpath(path, settings.MEDIA_ROOT).replace('\\', '/')
	return f"{settings.MEDIA_URL}{relative}"
