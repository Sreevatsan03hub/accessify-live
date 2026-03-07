/**
 * AudioWorkletProcessor — runs in the audio thread
 * Buffers incoming PCM samples and sends chunks to the main thread
 * every ~256ms (4096 samples @ 16kHz)
 */
class AudioCaptureProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this._buffer = [];
        this._bufferSize = 4096; // samples (~256ms at 16kHz)
    }

    process(inputs) {
        const input = inputs[0];
        if (!input || !input[0]) return true;

        const samples = input[0]; // Float32Array, mono channel
        for (let i = 0; i < samples.length; i++) {
            this._buffer.push(samples[i]);
        }

        if (this._buffer.length >= this._bufferSize) {
            const chunk = new Float32Array(this._buffer.splice(0, this._bufferSize));
            // Transfer ownership for zero-copy
            this.port.postMessage({ chunk: chunk.buffer }, [chunk.buffer]);
        }

        return true; // keep processor alive
    }
}

registerProcessor('audio-capture-processor', AudioCaptureProcessor);
