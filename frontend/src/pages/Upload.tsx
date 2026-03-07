import { useState, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { Select } from '../components/ui/Select'
import { Upload as UploadIcon, CheckCircle, AlertCircle } from 'lucide-react'

export default function Upload() {
  const [file, setFile] = useState<File | null>(null)
  const [language, setLanguage] = useState('en')
  const [translateTo, setTranslateTo] = useState('')
  const [progress, setProgress] = useState(0)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const navigate = useNavigate()

  const handleFileSelect = (selectedFile: File) => {
    const validTypes = ['video/mp4', 'video/x-matroska', 'video/x-msvideo', 'video/quicktime', 'video/webm']
    if (!validTypes.includes(selectedFile.type)) {
      setError('Invalid file type. Please upload MP4, MKV, AVI, MOV, or WebM video.')
      return
    }

    if (selectedFile.size > 500 * 1024 * 1024) {
      setError('File is too large. Maximum size is 500MB.')
      return
    }

    setFile(selectedFile)
    setError('')
    setSuccess(false)
  }

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.currentTarget.classList.add('border-accent')
  }

  const handleDragLeave = (e: React.DragEvent) => {
    e.currentTarget.classList.remove('border-accent')
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.currentTarget.classList.remove('border-accent')
    if (e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0])
    }
  }

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file')
      return
    }

    setUploading(true)
    setError('')

    try {
      // Simulate upload with progress
      for (let i = 0; i <= 100; i += 10) {
        await new Promise(resolve => setTimeout(resolve, 200))
        setProgress(i)
      }

      setSuccess(true)
      setProgress(100)

      // Simulate processing delay then redirect
      await new Promise(resolve => setTimeout(resolve, 2000))

      navigate('/player', {
        state: {
          sessionId: 'demo_' + Date.now(),
          filename: file.name,
          duration: 45.2,
          captions: [
            {
              id: 1,
              text: 'Welcome to the lecture',
              simplified_text: 'Welcome',
              keywords: [{ keyword: 'lecture', emoji: '📘' }],
              tone: { emotion: 'positive', intent: 'statement', emoji: '😊' },
              translation: null,
              sound_event: null,
              timestamp: 0,
            },
          ],
          vtt: 'WEBVTT\n\n1\n00:00:01.000 --> 00:00:05.000\nWelcome to the lecture',
        },
      })
    } catch (err) {
      setError('Upload failed. Please try again.')
      console.error('Upload error:', err)
    } finally {
      setUploading(false)
    }
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Math.round((bytes / Math.pow(k, i)) * 100) / 100 + ' ' + sizes[i]
  }

  return (
    <div className="max-w-2xl mx-auto px-4 py-12">
      <h1 className="text-4xl font-bold mb-2">Upload Video for Captioning</h1>
      <p className="text-muted mb-8">Upload a video file to automatically generate captions and translations</p>

      <div className="space-y-6">
        {/* File Upload Area */}
        <Card
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className="border-2 border-dashed cursor-pointer hover:border-accent transition-colors p-12 text-center"
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={(e) => e.target.files && handleFileSelect(e.target.files[0])}
            className="hidden"
          />

          {!file ? (
            <>
              <UploadIcon className="w-16 h-16 mx-auto mb-4 text-muted" />
              <h3 className="text-xl font-bold mb-2">Drop video here or click to select</h3>
              <p className="text-muted text-sm">Supported: MP4, MKV, AVI, MOV, WebM (max 500MB)</p>
            </>
          ) : (
            <>
              <CheckCircle className="w-16 h-16 mx-auto mb-4 text-accent" />
              <h3 className="text-xl font-bold mb-2">{file.name}</h3>
              <p className="text-muted text-sm">{formatFileSize(file.size)}</p>
            </>
          )}
        </Card>

        {/* Options */}
        {file && !uploading && (
          <>
            <Select
              label="Source Language"
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              options={[
                { value: 'en', label: 'English' },
                { value: 'hi', label: 'हिंदी (Hindi)' },
                { value: 'ta', label: 'தமிழ் (Tamil)' },
                { value: 'te', label: 'తెలుగు (Telugu)' },
              ]}
            />

            <Select
              label="Translate to (Optional)"
              value={translateTo}
              onChange={(e) => setTranslateTo(e.target.value)}
              options={[
                { value: '', label: 'No translation' },
                { value: 'hi', label: 'हिंदी (Hindi)' },
                { value: 'ta', label: 'தமிழ் (Tamil)' },
                { value: 'te', label: 'తెలుగు (Telugu)' },
              ]}
            />
          </>
        )}

        {/* Progress Bar */}
        {uploading && (
          <Card>
            <p className="text-sm font-semibold mb-3">Processing: {progress}%</p>
            <div className="w-full h-2 bg-black/40 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-primary to-accent transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          </Card>
        )}

        {/* Status Messages */}
        {error && (
          <div className="p-4 bg-warning/20 border border-warning rounded-lg text-warning flex gap-3">
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <p>{error}</p>
          </div>
        )}

        {success && (
          <div className="p-4 bg-accent/20 border border-accent rounded-lg text-accent flex gap-3">
            <CheckCircle className="w-5 h-5 flex-shrink-0" />
            <p>Video uploaded successfully! Redirecting to player...</p>
          </div>
        )}

        {/* Action Buttons */}
        {file && !uploading && (
          <div className="flex gap-4">
            <Button
              size="lg"
              className="flex-1"
              onClick={handleUpload}
            >
              <UploadIcon size={20} />
              Upload & Process
            </Button>
            <Button
              variant="ghost"
              size="lg"
              onClick={() => {
                setFile(null)
                setProgress(0)
              }}
            >
              Cancel
            </Button>
          </div>
        )}

        {!file && (
          <Button
            size="lg"
            className="w-full"
            onClick={() => fileInputRef.current?.click()}
          >
            <UploadIcon size={20} />
            Select Video
          </Button>
        )}
      </div>

      {/* Info Box */}
      <Card className="mt-8 border-accent/50 bg-accent/5">
        <h3 className="font-bold mb-3">What happens next:</h3>
        <ol className="space-y-2 text-sm text-muted list-decimal list-inside">
          <li>Your video is uploaded securely to our servers</li>
          <li>AI processes the audio to generate accurate captions</li>
          <li>Keywords are automatically identified and highlighted</li>
          <li>Optional translations are generated for your language</li>
          <li>Captions are saved and can be downloaded in multiple formats</li>
        </ol>
      </Card>
    </div>
  )
}
