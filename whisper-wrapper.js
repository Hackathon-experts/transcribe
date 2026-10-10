/*
 * whisper-wrapper.js
 * Simple wrapper around the whisper.cpp WASM build (main.js) + helpers.js.
 *
 *   const w = new WhisperWrapper({ uploadUrl: 'https://api.example.com/audio' });
 *   await w.init();
 *
 *   await w.startRecording();
 *   const r = await w.stopRecording();      // records -> transcribes -> uploads
 *   console.log(r.text);
 *
 *   const r2 = await w.importAudio(file);   // imports -> transcribes -> uploads
 *
 * Load order on the page:
 *   coi-serviceworker.js, helpers.js, whisper-wrapper.js
 * (main.js is loaded by init() itself.)
 */
(function (global) {
    'use strict';

    // globals that helpers.js' loadRemote() expects
    if (typeof global.dbName === 'undefined') global.dbName = 'whisper.ggerganov.com';
    if (typeof global.dbVersion === 'undefined') global.dbVersion = 1;

    const SAMPLE_RATE = 16000;
    const MODEL_FILE = 'whisper.bin'; // name inside the WASM virtual filesystem
    const SEGMENT_RE = /^\[(\d+):(\d+):(\d+)[.,](\d+)\s*-->\s*(\d+):(\d+):(\d+)[.,](\d+)\]\s*(.*)$/;

    function toSeconds(h, m, s, ms) {
        return (+h) * 3600 + (+m) * 60 + (+s) + (+ms) / 1000;
    }

    class WhisperWrapper {
        /**
         * @param {Object} opts
         * @param {string} [opts.engineUrl='main.js']   whisper WASM glue script
         * @param {string} [opts.modelUrl]              ggml tiny.en model URL
         * @param {number} [opts.modelSizeMB=75]
         * @param {string} [opts.language='en']
         * @param {number} [opts.threads]               default: min(8, cores/2)
         * @param {number} [opts.maxSeconds=300]        audio longer than this is truncated for transcription
         * @param {string} [opts.uploadUrl]             your API endpoint (omit to disable uploading)
         * @param {Object} [opts.uploadHeaders]         e.g. { Authorization: 'Bearer ...' }
         * @param {Object} [opts.uploadFields]          extra form fields sent with every upload
         * @param {string} [opts.uploadFieldName='audio']
         * @param {Function} [opts.uploadFn]            custom (payload) => Promise, replaces the default fetch upload
         * @param {Function} [opts.onStatus]            (state, message)
         * @param {Function} [opts.onProgress]          (0..1) model download progress
         * @param {Function} [opts.onSegment]           ({start, end, text}) as each segment is decoded
         * @param {Function} [opts.onLog]               (line) raw engine / debug output
         */
        constructor(opts = {}) {
            this.opts = Object.assign({
                engineUrl: 'main.js',
                modelUrl: 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin',
                modelSizeMB: 75,
                language: 'en',
                threads: Math.max(1, Math.min(8, Math.floor((navigator.hardwareConcurrency || 4) / 2))),
                maxSeconds: 300,
                uploadUrl: null,
                uploadHeaders: {},
                uploadFields: {},
                uploadFieldName: 'audio',
                uploadFn: null,
                onStatus: () => {},
                onProgress: () => {},
                onSegment: () => {},
                onLog: () => {},
            }, opts);

            this.ready = false;
            this.busy = false;
            this._instance = 0;
            this._segments = null;
            this._doneCb = null;
            this._rec = null;
        }

        // ------------------------------------------------------------------
        // Setup
        // ------------------------------------------------------------------

        /** Load the engine + model (model is cached in IndexedDB after first download). */
        async init() {
            if (this.ready) return;

            if (!global.crossOriginIsolated) {
                throw new Error(
                    'Page is not cross-origin isolated (SharedArrayBuffer unavailable). ' +
                    'Include coi-serviceworker.js before this script, serve over HTTPS or localhost, and reload.'
                );
            }
            if (typeof global.loadRemote !== 'function') {
                throw new Error('helpers.js must be loaded before whisper-wrapper.js');
            }

            this._status('loading', 'Loading engine...');
            await this._loadEngine();

            this._status('loading', 'Loading model...');
            const data = await new Promise((resolve, reject) => {
                global.loadRemote(
                    this.opts.modelUrl,
                    MODEL_FILE,
                    this.opts.modelSizeMB,
                    (p) => this.opts.onProgress(p),
                    (dst, buf) => resolve(buf),
                    () => reject(new Error('Model load cancelled or failed')),
                    (line) => this.opts.onLog(line)
                );
            });

            try { global.Module.FS_unlink(MODEL_FILE); } catch (e) { /* not there yet */ }
            global.Module.FS_createDataFile('/', MODEL_FILE, data, true, true);

            this._instance = global.Module.init(MODEL_FILE);
            if (!this._instance) throw new Error('Whisper failed to initialise the model');

            this.ready = true;
            this._status('ready', 'Ready');
        }

        /** Free the model instance. */
        dispose() {
            if (this._instance) {
                global.Module.free(this._instance);
                this._instance = 0;
            }
            this.ready = false;
        }

        _loadEngine() {
            return new Promise((resolve, reject) => {
                // Must be defined before main.js runs
                global.Module = {
                    print: (...a) => this._onPrint(a.join(' ')),
                    printErr: (...a) => this._onPrintErr(a.join(' ')),
                    setStatus: () => {},
                    onRuntimeInitialized: () => resolve(),
                };
                const s = document.createElement('script');
                // absolute URL: the pthread workers re-use this script's URL
                s.src = new URL(this.opts.engineUrl, location.href).href;
                s.onerror = () => reject(new Error('Failed to load ' + this.opts.engineUrl));
                document.head.appendChild(s);
            });
        }

        _onPrint(line) {
            this.opts.onLog(line);
            const m = SEGMENT_RE.exec(line.trim());
            if (m && this._segments) {
                const seg = {
                    start: toSeconds(m[1], m[2], m[3], m[4]),
                    end: toSeconds(m[5], m[6], m[7], m[8]),
                    text: m[9].trim(),
                };
                this._segments.push(seg);
                this.opts.onSegment(seg);
            }
        }

        _onPrintErr(line) {
            this.opts.onLog(line);
            // whisper_print_timings() runs after whisper_full() finishes in the worker thread
            if (this._doneCb && /total time\s*=/.test(line)) {
                const cb = this._doneCb;
                this._doneCb = null;
                setTimeout(cb, 50); // let any trailing stdout lines arrive first
            }
        }

        // ------------------------------------------------------------------
        // Recording
        // ------------------------------------------------------------------

        get isRecording() { return !!this._rec; }

        /** Ask for the mic and start recording. */
        async startRecording() {
            if (this._rec) throw new Error('Already recording');
            if (!navigator.mediaDevices || !global.MediaRecorder) {
                throw new Error('Recording is not supported in this browser');
            }

            const stream = await navigator.mediaDevices.getUserMedia({
                audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
            });

            const mimeType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', 'audio/ogg;codecs=opus']
                .find((t) => MediaRecorder.isTypeSupported(t));
            const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
            const chunks = [];
            recorder.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };

            this._rec = { stream, recorder, chunks, startedAt: Date.now() };
            recorder.start(250);
            this._status('recording', 'Recording...');
        }

        /** Seconds recorded so far (0 if not recording). */
        get recordingSeconds() {
            return this._rec ? (Date.now() - this._rec.startedAt) / 1000 : 0;
        }

        /** Stop recording without transcribing or uploading. Returns the audio Blob. */
        async stopRecordingRaw() {
            if (!this._rec) throw new Error('Not recording');
            const { stream, recorder, chunks } = this._rec;

            const blob = await new Promise((resolve) => {
                recorder.onstop = () => resolve(new Blob(chunks, { type: recorder.mimeType || 'audio/webm' }));
                recorder.stop();
            });
            stream.getTracks().forEach((t) => t.stop());
            this._rec = null;
            this._status('idle', 'Recording stopped');
            return blob;
        }

        /** Throw away the current recording. */
        async cancelRecording() {
            if (!this._rec) return;
            await this.stopRecordingRaw();
        }

        /**
         * Stop recording, then transcribe and upload.
         * @param {{transcribe?: boolean, upload?: boolean}} [o]
         */
        async stopRecording(o = {}) {
            const blob = await this.stopRecordingRaw();
            return this._process(blob, 'recording', o);
        }

        // ------------------------------------------------------------------
        // Import
        // ------------------------------------------------------------------

        /**
         * Transcribe and upload an audio file (File/Blob from <input type="file">).
         * @param {File|Blob} file
         * @param {{transcribe?: boolean, upload?: boolean}} [o]
         */
        async importAudio(file, o = {}) {
            return this._process(file, 'import', o);
        }

        // ------------------------------------------------------------------
        // Transcription only (no upload)
        // ------------------------------------------------------------------

        /**
         * @param {Blob|File|ArrayBuffer|Float32Array} input  Float32Array must be 16 kHz mono.
         * @returns {Promise<{text: string, segments: Array, duration: number}>}
         */
        async transcribe(input) {
            if (!this.ready) throw new Error('Call init() first');
            if (this.busy) throw new Error('Already transcribing');
            this.busy = true;
            try {
                let audio;
                if (input instanceof Float32Array) {
                    audio = input;
                } else {
                    this._status('decoding', 'Decoding audio...');
                    const buf = input instanceof ArrayBuffer ? input : await input.arrayBuffer();
                    audio = await this._decode(buf);
                }
                const duration = audio.length / SAMPLE_RATE;

                const max = this.opts.maxSeconds * SAMPLE_RATE;
                if (this.opts.maxSeconds > 0 && audio.length > max) {
                    this.opts.onLog('Audio is ' + duration.toFixed(0) + 's, truncating to ' + this.opts.maxSeconds + 's for transcription');
                    audio = audio.slice(0, max);
                }

                this._status('transcribing', 'Transcribing...');
                const segments = await this._run(audio);
                const text = segments
                    .map((s) => s.text)
                    .join(' ')
                    .replace(/\[BLANK_AUDIO\]/g, '')
                    .replace(/\s+/g, ' ')
                    .trim();

                this._status('ready', 'Done');
                return { text, segments, duration };
            } finally {
                this.busy = false;
            }
        }

        _run(audio) {
            return new Promise((resolve, reject) => {
                this._segments = [];
                this._doneCb = () => {
                    const segs = this._segments;
                    this._segments = null;
                    resolve(segs);
                };
                const rc = global.Module.full_default(
                    this._instance, audio, this.opts.language, this.opts.threads, false
                );
                if (rc !== 0) {
                    this._doneCb = null;
                    this._segments = null;
                    reject(new Error('Whisper failed to start (code ' + rc + ')'));
                }
            });
        }

        /** Decode any browser-supported audio to 16 kHz mono Float32Array. */
        async _decode(arrayBuffer) {
            // decodeAudioData resamples to the context's rate
            const ctx = new (global.OfflineAudioContext || global.webkitOfflineAudioContext)(1, 1, SAMPLE_RATE);
            let decoded;
            try {
                decoded = await ctx.decodeAudioData(arrayBuffer.slice(0));
            } catch (e) {
                throw new Error('Could not decode this audio format in your browser');
            }
            const ch = decoded.numberOfChannels;
            if (ch === 1) return new Float32Array(decoded.getChannelData(0));
            const out = new Float32Array(decoded.length);
            for (let c = 0; c < ch; c++) {
                const d = decoded.getChannelData(c);
                for (let i = 0; i < out.length; i++) out[i] += d[i] / ch;
            }
            return out;
        }

        // ------------------------------------------------------------------
        // Upload
        // ------------------------------------------------------------------

        /**
         * Upload audio (and optional transcript) to your API as multipart/form-data.
         * Fields sent: <uploadFieldName> (file), source, duration, transcript, segments (JSON), + uploadFields
         */
        async upload(payload) {
            if (this.opts.uploadFn) return this.opts.uploadFn(payload);
            if (!this.opts.uploadUrl) throw new Error('uploadUrl is not set');

            const fd = new FormData();
            fd.append(this.opts.uploadFieldName, payload.blob, payload.filename);
            fd.append('source', payload.source);
            if (payload.duration != null) fd.append('duration', String(payload.duration));
            if (payload.text != null) fd.append('transcript', payload.text);
            if (payload.segments) fd.append('segments', JSON.stringify(payload.segments));
            for (const [k, v] of Object.entries(this.opts.uploadFields)) fd.append(k, v);

            const res = await fetch(this.opts.uploadUrl, {
                method: 'POST',
                headers: this.opts.uploadHeaders, // don't set Content-Type: the browser adds the boundary
                body: fd,
            });
            if (!res.ok) throw new Error('Upload failed: HTTP ' + res.status);

            const ct = res.headers.get('content-type') || '';
            return ct.includes('json') ? res.json() : res.text();
        }

        // ------------------------------------------------------------------
        // Shared pipeline
        // ------------------------------------------------------------------

        async _process(blob, source, o = {}) {
            const doTranscribe = o.transcribe !== false;
            const doUpload = o.upload !== false && !!(this.opts.uploadUrl || this.opts.uploadFn);

            const result = {
                source,
                blob,
                filename: blob.name || ('recording-' + new Date().toISOString().replace(/[:.]/g, '-') + '.' + this._ext(blob.type)),
                text: null,
                segments: null,
                duration: null,
                transcriptionError: null,
                upload: null,
            };

            if (doTranscribe) {
                try {
                    const t = await this.transcribe(blob);
                    result.text = t.text;
                    result.segments = t.segments;
                    result.duration = t.duration;
                } catch (e) {
                    result.transcriptionError = e.message;
                    this.opts.onLog('Transcription error: ' + e.message);
                }
            }

            // Audio is uploaded even if transcription failed
            if (doUpload) {
                this._status('uploading', 'Uploading...');
                try {
                    const response = await this.upload(result);
                    result.upload = { ok: true, response };
                } catch (e) {
                    result.upload = { ok: false, error: e.message };
                }
            }

            this._status('ready', 'Done');
            return result;
        }

        _ext(mime) {
            if (!mime) return 'webm';
            if (mime.includes('mp4')) return 'm4a';
            if (mime.includes('ogg')) return 'ogg';
            if (mime.includes('wav')) return 'wav';
            if (mime.includes('mpeg')) return 'mp3';
            return 'webm';
        }

        _status(state, msg) {
            this.opts.onStatus(state, msg);
        }
    }

    global.WhisperWrapper = WhisperWrapper;
})(window);
