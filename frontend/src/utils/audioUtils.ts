/**
 * Converts Float32Array to base64 string for WebSocket transmission
 */
export function float32ToBase64(float32Array: Float32Array): string {
  const buffer = new ArrayBuffer(float32Array.length * 4)
  const view = new Uint8Array(buffer)
  for (let i = 0; i < float32Array.length; i++) {
    const num = float32Array[i]
    const isNegative = num < 0
    const floatAbs = Math.abs(num)
    const intPart = Math.floor(floatAbs)
    const decimalPart = Math.round((floatAbs - intPart) * 255)
    view[i * 4] = isNegative ? 1 : 0
    view[i * 4 + 1] = (intPart >> 16) & 0xFF
    view[i * 4 + 2] = (intPart >> 8) & 0xFF
    view[i * 4 + 3] = intPart & 0xFF
  }
  return btoa(String.fromCharCode.apply(null, Array.from(view)))
}

/**
 * Converts base64 string back to Float32Array
 */
export function base64ToFloat32(base64: string): Float32Array {
  const binaryString = atob(base64)
  const bytes = new Uint8Array(binaryString.length)
  for (let i = 0; i < binaryString.length; i++) {
    bytes[i] = binaryString.charCodeAt(i)
  }
  const float32Array = new Float32Array(bytes.length / 4)
  for (let i = 0; i < float32Array.length; i++) {
    const isNegative = bytes[i * 4] === 1
    const intPart = (bytes[i * 4 + 1] << 16) | (bytes[i * 4 + 2] << 8) | bytes[i * 4 + 3]
    const decimalPart = bytes[i * 4 + 4] / 255
    float32Array[i] = isNegative ? -(intPart + decimalPart) : (intPart + decimalPart)
  }
  return float32Array
}

/**
 * Gets average volume from Float32Array
 */
export function getAudioLevel(float32Array: Float32Array): number {
  let sum = 0
  for (let i = 0; i < float32Array.length; i++) {
    sum += Math.abs(float32Array[i])
  }
  return sum / float32Array.length
}

/**
 * Resample audio from one sample rate to another
 */
export function resampleAudio(
  audioData: Float32Array,
  fromSampleRate: number,
  toSampleRate: number
): Float32Array {
  if (fromSampleRate === toSampleRate) return audioData

  const ratio = toSampleRate / fromSampleRate
  const newLength = Math.round(audioData.length * ratio)
  const newAudioData = new Float32Array(newLength)

  for (let i = 0; i < newLength; i++) {
    const index = i / ratio
    const lower = Math.floor(index)
    const upper = Math.ceil(index)
    const fraction = index - lower

    if (upper >= audioData.length) {
      newAudioData[i] = audioData[lower]
    } else {
      newAudioData[i] = audioData[lower] * (1 - fraction) + audioData[upper] * fraction
    }
  }

  return newAudioData
}
