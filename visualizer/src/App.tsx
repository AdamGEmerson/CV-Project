import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import Plot from 'react-plotly.js'
import type { Config, Data, Layout } from 'plotly.js-basic-dist'
import './App.css'
const CLUSTER_COLOR_PALETTE = [
  '#0f172a',
  '#2563eb',
  '#dc2626',
  '#059669',
  '#db2777',
  '#f97316',
  '#9333ea',
  '#14b8a6',
  '#facc15',
  '#94a3b8'
]

const HAND_COLORS = ['#2563eb', '#f97316', '#0ea5e9', '#9333ea']

// Finger colors for landmarks
const FINGER_COLORS = {
  wrist: '#94a3b8', // Gray
  thumb: '#dc2626', // Red
  index: '#2563eb', // Blue
  middle: '#059669', // Green
  ring: '#9333ea', // Purple
  pinky: '#f97316' // Orange
}

// Finger landmark ranges (inclusive)
const FINGER_LANDMARKS = {
  wrist: [0],
  thumb: [1, 2, 3, 4],
  index: [5, 6, 7, 8],
  middle: [9, 10, 11, 12],
  ring: [13, 14, 15, 16],
  pinky: [17, 18, 19, 20]
}

const FINGER_LABELS = {
  wrist: 'Wrist',
  thumb: 'Thumb',
  index: 'Index',
  middle: 'Middle',
  ring: 'Ring',
  pinky: 'Pinky'
}

// MediaPipe hand landmark connections (21 points)
// Format: [from_index, to_index]
const HAND_CONNECTIONS = [
  // Wrist to finger bases
  [0, 1], [0, 5], [0, 9], [0, 13], [0, 17],
  // Thumb
  [1, 2], [2, 3], [3, 4],
  // Index finger
  [5, 6], [6, 7], [7, 8],
  // Middle finger
  [9, 10], [10, 11], [11, 12],
  // Ring finger
  [13, 14], [14, 15], [15, 16],
  // Pinky
  [17, 18], [18, 19], [19, 20],
  // Thumb to index connection
  [2, 5]
]

type Landmark = [number, number, number] // [x, y, z]

type ClusterHandLandmarks = {
  hand_index: number
  landmarks: Landmark[] // 21 landmarks
  count: number
}

type ClusterLandmarksSummary = {
  hands: ClusterHandLandmarks[]
}

type ClusteredFrame = {
  segment: number
  frame: number
  feature_idx: number
  cluster: number
  x?: number
  y?: number
  z?: number
  hand_landmarks?: Landmark[][] // Array of hands, each with 21 landmarks
}

type ClusterData = {
  n_segments: number
  n_frames: number
  n_clusters: number
  n_noise: number
  successful_segments: number[]
  failed_segments: number[]
  clustered_frames: ClusteredFrame[]
  cluster_distribution: Record<string, number>
  cluster_landmarks?: Record<string, ClusterLandmarksSummary>
}

const DATA_PATH = '/all_segments_clustered_with_xy.json'
const DEFAULT_FPS = 25

type ClusterStat = {
  id: number
  label: string
  count: number
  percent: number
}

type HoverPreviewState = {
  frame: ClusteredFrame
  status: 'loading' | 'ready' | 'error'
  imageUrl?: string
  error?: string
}

const padSegment = (segment: number) => segment.toString().padStart(3, '0')
const buildSegmentVideoPath = (segment: number) => `/segments/segment_${padSegment(segment)}.mp4`

function App() {
  const [data, setData] = useState<ClusterData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null)
  const [selectedSegment, setSelectedSegment] = useState<number | null>(null)
  const [selectedFrame, setSelectedFrame] = useState<ClusteredFrame | null>(null)
  const [fps, setFps] = useState(DEFAULT_FPS)
  const [pendingSeek, setPendingSeek] = useState<number | null>(null)
  const [hoverPreview, setHoverPreview] = useState<HoverPreviewState | null>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const frameImageCache = useRef<Map<string, string>>(new Map())
  const hoverRequestId = useRef(0)

  useEffect(() => {
    fetch(DATA_PATH)
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`Unable to load cluster data (${response.status})`)
        }
        return response.json() as Promise<ClusterData>
      })
      .then((payload) => {
        setData(payload)
        if (payload.cluster_distribution) {
          const primaryClusterId = Object.entries(payload.cluster_distribution)
            .filter(([clusterId]) => clusterId !== '-1')
            .sort(([, countA], [, countB]) => Number(countB) - Number(countA))[0]?.[0]
          if (primaryClusterId) {
            setSelectedCluster(Number(primaryClusterId))
          }
        }
      })
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    frameImageCache.current.clear()
  }, [fps])

  const clustersById = useMemo(() => {
    const map = new Map<number, ClusteredFrame[]>()
    if (!data) return map
    for (const frame of data.clustered_frames) {
      if (!map.has(frame.cluster)) {
        map.set(frame.cluster, [])
      }
      map.get(frame.cluster)!.push(frame)
    }
    for (const clusterFrames of map.values()) {
      clusterFrames.sort((a, b) => {
        if (a.segment === b.segment) {
          return a.frame - b.frame
        }
        return a.segment - b.segment
      })
    }
    return map
  }, [data])

  const clusterLandmarksMap = useMemo(() => {
    const map = new Map<number, ClusterLandmarksSummary>()
    if (!data?.cluster_landmarks) {
      return map
    }
    Object.entries(data.cluster_landmarks).forEach(([clusterId, summary]) => {
      map.set(Number(clusterId), summary)
    })
    return map
  }, [data])

  const clusterStats: ClusterStat[] = useMemo(() => {
    if (!data) return []
    const entries = Object.entries(data.cluster_distribution)
    const stats = entries.map(([clusterId, count]) => {
      const numericId = Number(clusterId)
      const label = numericId === -1 ? 'Noise' : `Cluster ${clusterId}`
      const percent = (Number(count) / data.n_frames) * 100
      return { id: numericId, label, count: Number(count), percent }
    })
    stats.sort((a, b) => {
      if (a.id === -1) return 1
      if (b.id === -1) return -1
      return b.count - a.count
    })
    return stats
  }, [data])

  const selectedClusterLandmarks = useMemo(() => {
    if (selectedCluster === null) return null
    return clusterLandmarksMap.get(selectedCluster) ?? null
  }, [clusterLandmarksMap, selectedCluster])

  // Prepare 3D hand skeleton data for Plotly - one plot per hand
  const handSkeletonPlots = useMemo(() => {
    if (!selectedClusterLandmarks || selectedClusterLandmarks.hands.length === 0) {
      return []
    }

    return selectedClusterLandmarks.hands.map((handData) => {
      const landmarks = handData.landmarks
      const color = HAND_COLORS[handData.hand_index % HAND_COLORS.length]

      // Extract x, y, z coordinates for all landmarks
      const xs = landmarks.map((lm) => lm[0])
      const ys = landmarks.map((lm) => lm[1])
      const zs = landmarks.map((lm) => lm[2])

      // Calculate bounding box for camera positioning
      const minX = Math.min(...xs)
      const maxX = Math.max(...xs)
      const minY = Math.min(...ys)
      const maxY = Math.max(...ys)
      const minZ = Math.min(...zs)
      const maxZ = Math.max(...zs)
      const centerX = (minX + maxX) / 2
      const centerY = (minY + maxY) / 2
      const centerZ = (minZ + maxZ) / 2
      // Very tight range for maximum zoom (minimal padding, 0.5x)
      const rawRange = Math.max(maxX - minX, maxY - minY, maxZ - minZ)
      const range = rawRange > 0 ? rawRange * 0.5 : 0.5

      // Calculate camera position to view with thumb pointing up
      // Position camera very close for maximum zoom (0.6x distance)
      const cameraDistance = range * 0.6
      const cameraX = centerX + cameraDistance * 0.7
      const cameraY = centerY + cameraDistance * 0.3
      const cameraZ = centerZ + cameraDistance * 0.5

      const traces: Partial<Data>[] = []

      // Create scatter plots for each finger with different colors
      Object.entries(FINGER_LANDMARKS).forEach(([fingerName, indices]) => {
        const fingerXs: number[] = []
        const fingerYs: number[] = []
        const fingerZs: number[] = []
        const fingerLabels: string[] = []

        indices.forEach((idx) => {
          if (idx < landmarks.length) {
            fingerXs.push(landmarks[idx][0])
            fingerYs.push(landmarks[idx][1])
            fingerZs.push(landmarks[idx][2])
            // Label fingertips and key joints
            if (idx === 0) {
              fingerLabels.push('Wrist')
            } else if (idx === 4) {
              fingerLabels.push('Thumb tip')
            } else if (idx === 8) {
              fingerLabels.push('Index tip')
            } else if (idx === 12) {
              fingerLabels.push('Middle tip')
            } else if (idx === 16) {
              fingerLabels.push('Ring tip')
            } else if (idx === 20) {
              fingerLabels.push('Pinky tip')
            } else {
              fingerLabels.push('')
            }
          }
        })

        if (fingerXs.length > 0) {
          traces.push({
            type: 'scatter3d',
            mode: 'text+markers',
            name: FINGER_LABELS[fingerName as keyof typeof FINGER_LABELS],
            x: fingerXs,
            y: fingerYs,
            z: fingerZs,
            text: fingerLabels,
            textposition: 'top center',
            textfont: {
              size: 10,
              color: FINGER_COLORS[fingerName as keyof typeof FINGER_COLORS]
            },
            marker: {
              size: fingerName === 'wrist' ? 10 : 8,
              color: FINGER_COLORS[fingerName as keyof typeof FINGER_COLORS],
              opacity: 0.9,
              line: {
                color: '#ffffff',
                width: 1
              }
            },
            showlegend: true
          })
        }
      })

      // Create line traces for connections (bones) - color by finger
      HAND_CONNECTIONS.forEach(([fromIdx, toIdx]) => {
        if (fromIdx < landmarks.length && toIdx < landmarks.length) {
          // Determine which finger this connection belongs to
          let connectionColor = color
          if (FINGER_LANDMARKS.thumb.includes(fromIdx) || FINGER_LANDMARKS.thumb.includes(toIdx)) {
            connectionColor = FINGER_COLORS.thumb
          } else if (FINGER_LANDMARKS.index.includes(fromIdx) || FINGER_LANDMARKS.index.includes(toIdx)) {
            connectionColor = FINGER_COLORS.index
          } else if (FINGER_LANDMARKS.middle.includes(fromIdx) || FINGER_LANDMARKS.middle.includes(toIdx)) {
            connectionColor = FINGER_COLORS.middle
          } else if (FINGER_LANDMARKS.ring.includes(fromIdx) || FINGER_LANDMARKS.ring.includes(toIdx)) {
            connectionColor = FINGER_COLORS.ring
          } else if (FINGER_LANDMARKS.pinky.includes(fromIdx) || FINGER_LANDMARKS.pinky.includes(toIdx)) {
            connectionColor = FINGER_COLORS.pinky
          } else {
            connectionColor = FINGER_COLORS.wrist
          }

          traces.push({
            type: 'scatter3d',
            mode: 'lines',
            name: '',
            x: [landmarks[fromIdx][0], landmarks[toIdx][0]],
            y: [landmarks[fromIdx][1], landmarks[toIdx][1]],
            z: [landmarks[fromIdx][2], landmarks[toIdx][2]],
            line: {
              color: connectionColor,
              width: 3
            },
            showlegend: false,
            hoverinfo: 'skip'
          })
        }
      })

      return {
        handIndex: handData.hand_index,
        traces,
        layout: {
          margin: { l: 0, r: 0, b: 0, t: 0 },
          scene: {
            xaxis: { 
              title: { text: 'X' }, 
              range: [centerX - range, centerX + range],
              autorange: false
            },
            yaxis: { 
              title: { text: 'Y' }, 
              range: [centerY - range, centerY + range],
              autorange: false
            },
            zaxis: { 
              title: { text: 'Z' }, 
              range: [centerZ - range, centerZ + range],
              autorange: false
            },
            aspectmode: 'cube',
            camera: {
              eye: { x: cameraX, y: cameraY, z: cameraZ },
              center: { x: centerX, y: centerY, z: centerZ },
              up: { x: 0, y: 1, z: 0 } // Y-axis up (thumb should point this way)
            }
          },
          hovermode: 'closest',
          paper_bgcolor: 'rgba(0,0,0,0)',
          plot_bgcolor: 'rgba(0,0,0,0)',
          showlegend: true,
          legend: {
            x: 0.02,
            y: 0.98,
            bgcolor: 'rgba(255,255,255,0.9)',
            bordercolor: '#cbd5f5',
            borderwidth: 1,
            font: { size: 11 }
          }
        },
        count: handData.count
      }
    })
  }, [selectedClusterLandmarks])

  const framesForSelectedCluster = useMemo(() => {
    if (selectedCluster === null) return []
    return clustersById.get(selectedCluster) ?? []
  }, [clustersById, selectedCluster])

  const allScatterFrames = useMemo(
    () =>
      data
        ? data.clustered_frames.filter(
            (frame) =>
              typeof frame.x === 'number' &&
              typeof frame.y === 'number' &&
              typeof frame.z === 'number'
          )
        : [],
    [data]
  )

  const frameByFeatureIdx = useMemo(() => {
    const map = new Map<number, ClusteredFrame>()
    allScatterFrames.forEach((frame) => {
      map.set(frame.feature_idx, frame)
    })
    return map
  }, [allScatterFrames])

  const groupedBySegment = useMemo(() => {
    const grouping = new Map<number, ClusteredFrame[]>()
    for (const frame of framesForSelectedCluster) {
      if (!grouping.has(frame.segment)) {
        grouping.set(frame.segment, [])
      }
      grouping.get(frame.segment)!.push(frame)
    }
    return Array.from(grouping.entries()).sort((a, b) => a[0] - b[0])
  }, [framesForSelectedCluster])

  useEffect(() => {
    hoverRequestId.current += 1
    setHoverPreview(null)
  }, [selectedCluster])

  const scatterData = useMemo(() => {
    if (allScatterFrames.length === 0) {
      return []
    }

    const grouped = new Map<number, ClusteredFrame[]>()
    allScatterFrames.forEach((frame) => {
      if (!grouped.has(frame.cluster)) {
        grouped.set(frame.cluster, [])
      }
      grouped.get(frame.cluster)!.push(frame)
    })

    const traces: Partial<Data>[] = []
    for (const [clusterId, frames] of grouped) {
      const paletteIndex =
        clusterId === -1
          ? CLUSTER_COLOR_PALETTE.length - 1
          : clusterId % (CLUSTER_COLOR_PALETTE.length - 1)
      const baseColor = CLUSTER_COLOR_PALETTE[Math.max(0, paletteIndex)]
      const isSelectedCluster =
        selectedCluster === null ||
        selectedCluster === clusterId ||
        (selectedCluster === -1 && clusterId === -1)

      const trace: Partial<Data> = {
        type: 'scatter3d',
        mode: 'markers',
        name: clusterId === -1 ? 'Noise' : `Cluster ${clusterId}`,
        x: frames.map((frame) => frame.x ?? 0),
        y: frames.map((frame) => frame.y ?? 0),
        z: frames.map((frame) => frame.z ?? 0),
        text: frames.map(
          (frame) =>
            `Segment ${padSegment(frame.segment)} • Frame ${frame.frame} • Cluster ${frame.cluster}`
        ),
        hoverinfo: 'text',
        customdata: frames.map((frame) => frame.feature_idx),
        marker: {
          size: isSelectedCluster ? 6 : 3.5,
          color: baseColor,
          opacity: isSelectedCluster ? 0.9 : 0.2
        }
      }
      traces.push(trace)
    }

    return traces
  }, [allScatterFrames, selectedCluster])

  const scatterLayout = useMemo<Partial<Layout>>(
    () => ({
      margin: { l: 0, r: 0, b: 0, t: 0 },
      scene: {
        xaxis: { title: { text: 'PC 1' } },
        yaxis: { title: { text: 'PC 2' } },
        zaxis: { title: { text: 'PC 3' } }
      },
      hovermode: 'closest',
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      showlegend: false
    }),
    []
  )

  const scatterConfig = useMemo<Partial<Config>>(
    () => ({
      displaylogo: false,
      responsive: true
    }),
    []
  )

  const captureFrameImage = useCallback(
    async (frame: ClusteredFrame) => {
      const cacheKey = `${frame.segment}-${frame.frame}`
      const cached = frameImageCache.current.get(cacheKey)
      if (cached) {
        return cached
      }
      if (!fps || fps <= 0) {
        throw new Error('Set FPS > 0 to capture frames')
      }
      const video = document.createElement('video')
      video.crossOrigin = 'anonymous'
      video.preload = 'auto'
      video.muted = true
      video.src = buildSegmentVideoPath(frame.segment)
      await new Promise<void>((resolve, reject) => {
        video.onloadedmetadata = () => {
          video.onloadedmetadata = null
          resolve()
        }
        video.onerror = () => {
          video.onerror = null
          reject(new Error('Unable to load video for preview'))
        }
      })
      const rawTime = frame.frame / fps
      if (!Number.isFinite(rawTime)) {
        throw new Error('Invalid frame timing')
      }
      const duration = Number.isFinite(video.duration) ? video.duration : 0
      const safeTarget =
        duration > 0 ? Math.min(rawTime, Math.max(duration - 0.02, 0)) : Math.max(rawTime, 0)
      await new Promise<void>((resolve, reject) => {
        const cleanup = () => {
          video.onseeked = null
          video.onerror = null
        }
        video.onseeked = () => {
          cleanup()
          resolve()
        }
        video.onerror = () => {
          cleanup()
          reject(new Error('Unable to seek to requested frame'))
        }
        video.currentTime = safeTarget
      })
      const canvas = document.createElement('canvas')
      const width = video.videoWidth || 640
      const height = video.videoHeight || 360
      canvas.width = width
      canvas.height = height
      const ctx = canvas.getContext('2d')
      if (!ctx) {
        throw new Error('Canvas context unavailable')
      }
      ctx.drawImage(video, 0, 0, width, height)
      const dataUrl = canvas.toDataURL('image/png')
      frameImageCache.current.set(cacheKey, dataUrl)
      video.src = ''
      return dataUrl
    },
    [fps]
  )

  const handlePointHover = useCallback(
    (frame: ClusteredFrame) => {
      const cacheKey = `${frame.segment}-${frame.frame}`
      const cached = frameImageCache.current.get(cacheKey)
      if (cached) {
        setHoverPreview({ frame, status: 'ready', imageUrl: cached })
        return
      }
      const requestId = ++hoverRequestId.current
      setHoverPreview({ frame, status: 'loading' })
      captureFrameImage(frame)
        .then((previewUrl) => {
          if (hoverRequestId.current !== requestId) return
          if (!previewUrl) {
            setHoverPreview({ frame, status: 'error', error: 'Unable to capture frame' })
            return
          }
          setHoverPreview({ frame, status: 'ready', imageUrl: previewUrl })
        })
        .catch((err: Error) => {
          if (hoverRequestId.current !== requestId) return
          setHoverPreview({ frame, status: 'error', error: err.message })
        })
    },
    [captureFrameImage]
  )

  const handleScatterLeave = useCallback(() => {
    hoverRequestId.current += 1
    setHoverPreview(null)
  }, [])

  const videoSrc = selectedSegment !== null ? buildSegmentVideoPath(selectedSegment) : null

  const seekTime = useMemo(() => {
    if (!selectedFrame || !fps) return null
    const offset = selectedFrame.frame / fps
    return Number.isFinite(offset) ? Math.max(offset - 0.3, 0) : null
  }, [selectedFrame, fps])

  useEffect(() => {
    if (seekTime === null) return
    const video = videoRef.current
    if (!video) return
    if (video.readyState >= 1) {
      video.currentTime = seekTime
      video.play().catch(() => undefined)
    } else {
      setPendingSeek(seekTime)
    }
  }, [seekTime, selectedSegment])

  const handleVideoLoaded = () => {
    if (pendingSeek === null) return
    const video = videoRef.current
    if (!video) return
    video.currentTime = pendingSeek
    video.play().catch(() => undefined)
    setPendingSeek(null)
  }

  const handleClusterChange = (clusterId: number) => {
    setSelectedCluster(clusterId)
    setSelectedSegment(null)
    setSelectedFrame(null)
  }

  const handleSegmentToggle = (segmentId: number) => {
    setSelectedSegment((current) => (current === segmentId ? null : segmentId))
    setSelectedFrame(null)
  }

  const handleFrameClick = (frame: ClusteredFrame) => {
    setSelectedSegment(frame.segment)
    setSelectedFrame(frame)
  }

  const resolveFrameFromEvent = useCallback(
    (event: Readonly<any>) => {
      const point = event?.points?.[0]
      if (!point) {
        return null
      }
      // Handle both direct customdata and array access
      const customdata = point.customdata ?? point.data?.customdata?.[point.pointNumber]
      if (customdata == null) {
        return null
      }
      const featureIdx = Array.isArray(customdata) ? customdata[0] : customdata
      const numericIdx = Number(featureIdx)
      if (Number.isNaN(numericIdx)) {
        return null
      }
      return frameByFeatureIdx.get(numericIdx) ?? null
    },
    [frameByFeatureIdx]
  )

  const handlePlotHover = useCallback(
    (event: Readonly<any>) => {
      const frame = resolveFrameFromEvent(event)
      if (frame) {
        handlePointHover(frame)
      }
    },
    [handlePointHover, resolveFrameFromEvent]
  )

  const handlePlotClick = useCallback(
    (event: Readonly<any>) => {
      const frame = resolveFrameFromEvent(event)
      if (frame) {
        handleFrameClick(frame)
      }
    },
    [handleFrameClick, resolveFrameFromEvent]
  )

  const handlePlotUnhover = useCallback(() => {
    handleScatterLeave()
  }, [handleScatterLeave])

  const scatterPlotElement = useMemo(
    () => (
      <Plot
        data={scatterData}
        layout={scatterLayout}
        config={scatterConfig}
        style={{ width: '100%', height: '100%' }}
        onHover={handlePlotHover}
        onUnhover={handlePlotUnhover}
        onClick={handlePlotClick}
      />
    ),
    [scatterConfig, scatterData, scatterLayout, handlePlotClick, handlePlotHover, handlePlotUnhover]
  )

  if (loading) {
    return (
      <div className="app">
        <p className="status">Loading cluster data…</p>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="app">
        <p className="status error">Failed to load cluster data: {error ?? 'Unknown error'}</p>
      </div>
    )
  }

  return (
    <div className="app">
      <header>
        <p className="eyebrow">Clay-hand pipeline · cluster explorer</p>
        <h1>Cluster visualizer</h1>
        <p className="subtitle">
          Browse {data.n_frames.toLocaleString()} clustered frames ({data.n_segments} segments) and
          jump straight into the source clips.
        </p>
      </header>

      <section className="summary-grid">
        <article>
          <p className="summary-label">Clusters</p>
          <p className="summary-value">{data.n_clusters}</p>
          <p className="summary-note">
            {data.n_noise.toLocaleString()} frames labelled as noise
          </p>
        </article>
        <article>
          <p className="summary-label">Segments covered</p>
          <p className="summary-value">{data.successful_segments.length}</p>
          <p className="summary-note">
            {data.failed_segments.length} segments failed in preprocessing
          </p>
        </article>
        <article>
          <p className="summary-label">FPS</p>
          <div className="fps-control">
            <input
              type="number"
              step="0.1"
              min="1"
              value={fps}
              onChange={(event) => setFps(Number(event.target.value) || DEFAULT_FPS)}
              aria-label="Frames per second"
            />
            <span>frames/sec</span>
          </div>
          <p className="summary-note">Used to convert frame indices to timestamps</p>
        </article>
      </section>

      <section>
        <div className="section-header">
          <h2>Clusters</h2>
          <p>Select a cluster to inspect its frames grouped by segment.</p>
        </div>
        <div className="cluster-list">
          {clusterStats.map((stat) => (
            <button
              key={stat.id}
              type="button"
              className={`cluster-chip ${
                stat.id === selectedCluster ? 'cluster-chip--active' : ''
              }`}
              onClick={() => handleClusterChange(stat.id)}
            >
              <span className="cluster-chip__label">{stat.label}</span>
              <span className="cluster-chip__count">{stat.count.toLocaleString()} frames</span>
              <span className="cluster-chip__percent">{stat.percent.toFixed(1)}%</span>
            </button>
          ))}
        </div>
      </section>

      <section className="scatter-panel">
        <div className="section-header">
          <h2>3D embedding explorer</h2>
          <p>Hover points in PCA space to inspect frames and click to jump into the clip.</p>
        </div>
        {allScatterFrames.length > 0 ? (
          <div className="scatter-content">
            <div className="scatter-plot-wrapper">{scatterPlotElement}</div>
            <div className="scatter-preview-card">
              {hoverPreview ? (
                <>
                  <p className="segment-label">
                    Segment {padSegment(hoverPreview.frame.segment)} · Frame{' '}
                    {hoverPreview.frame.frame}
                  </p>
                  <p className="segment-meta">
                    {(hoverPreview.frame.frame / fps).toFixed(2)}s · Cluster{' '}
                    {hoverPreview.frame.cluster}
                  </p>
                  {hoverPreview.status === 'loading' && (
                    <p className="info-note">Capturing frame…</p>
                  )}
                  {hoverPreview.status === 'error' && (
                    <p className="info-note info-note--error">
                      {hoverPreview.error ?? 'Unable to capture frame'}
                    </p>
                  )}
                  {hoverPreview.status === 'ready' && hoverPreview.imageUrl && (
                    <img
                      src={hoverPreview.imageUrl}
                      alt={`Frame ${hoverPreview.frame.frame} from segment ${padSegment(
                        hoverPreview.frame.segment
                      )}`}
                    />
                  )}
                </>
              ) : (
                <p className="info-note">Hover a point to generate a still frame preview.</p>
              )}
            </div>
          </div>
        ) : (
          <p className="empty-state">
            No embedding coordinates available for the selected cluster yet. Re-run clustering to
            enrich the dataset.
          </p>
        )}
      </section>

      <section className="hand-explorer-panel">
        <div className="section-header">
          <h2>3D Hand skeleton explorer</h2>
          <p>Average normalized hand pose reconstructed from all 21 landmarks per hand. Each finger is color-coded and labeled.</p>
        </div>
        {selectedCluster === null ? (
          <p className="empty-state">Select a cluster to visualize its hand skeleton.</p>
        ) : !selectedClusterLandmarks || selectedClusterLandmarks.hands.length === 0 ? (
          <p className="empty-state">No landmark data captured for this cluster yet.</p>
        ) : (
          <div className="hand-skeleton-grid">
            {handSkeletonPlots.map((plotData) => (
              <div key={plotData.handIndex} className="hand-skeleton-pane">
                <div className="hand-skeleton-header">
                  <span
                    className="hand-skeleton-swatch"
                    style={{
                      background: HAND_COLORS[plotData.handIndex % HAND_COLORS.length]
                    }}
                  />
                  <span className="hand-skeleton-title">
                    Hand {plotData.handIndex + 1} ({plotData.count.toLocaleString()} frames)
                  </span>
                </div>
                <div className="hand-skeleton-plot">
                  <Plot
                    data={plotData.traces}
                    layout={plotData.layout}
                    config={{
                      displaylogo: false,
                      responsive: true
                    }}
                    style={{ width: '100%', height: '450px' }}
                  />
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="details-layout">
        <div className="segment-panel">
          <div className="section-header">
            <h2>
              {selectedCluster === null
                ? 'Select a cluster'
                : selectedCluster === -1
                  ? 'Noise frames'
                  : `Cluster ${selectedCluster}`}
            </h2>
            <p>
              {framesForSelectedCluster.length === 0
                ? 'No frames available.'
                : `${framesForSelectedCluster.length.toLocaleString()} frames across ${
                    groupedBySegment.length
                  } segments.`}
            </p>
          </div>

          <div className="segment-panel-content">
            <div className="segment-groups-wrapper">
              <div className="segment-groups">
                {groupedBySegment.map(([segmentId, frames]) => (
                  <article
                    key={segmentId}
                    className={`segment-group ${
                      selectedSegment === segmentId ? 'segment-group--active' : ''
                    }`}
                  >
                    <header>
      <div>
                      <p className="segment-label">Segment {padSegment(segmentId)}</p>
                      <p className="segment-meta">
                        {frames.length} frame{frames.length === 1 ? '' : 's'}
                      </p>
      </div>
                    <button type="button" onClick={() => handleSegmentToggle(segmentId)}>
                      {selectedSegment === segmentId ? 'Collapse' : 'Preview'}
                    </button>
                    </header>
                    {selectedSegment === segmentId && (
                      <div className="frame-grid">
                        {frames.map((frame) => (
                          <button
                            key={frame.feature_idx}
                            type="button"
                            className={`frame-pill ${
                              selectedFrame?.feature_idx === frame.feature_idx ? 'frame-pill--active' : ''
                            }`}
                            onClick={() => handleFrameClick(frame)}
                          >
                            <span>Frame {frame.frame}</span>
                            <span>{(frame.frame / fps).toFixed(2)}s</span>
        </button>
                        ))}
                      </div>
                    )}
                  </article>
                ))}
                {groupedBySegment.length === 0 && (
                  <p className="empty-state">Pick a cluster to see its frames.</p>
                )}
              </div>
            </div>

            <div className="player-panel">
              <div className="section-header">
                <h2>Preview</h2>
                <p>
                  {selectedSegment === null
                    ? 'Select a segment/frame to load the corresponding clip.'
                    : `segment_${padSegment(selectedSegment)}.mp4`}
        </p>
      </div>
              {videoSrc ? (
                <>
                  <video
                    key={videoSrc}
                    ref={videoRef}
                    src={videoSrc}
                    controls
                    playsInline
                    onLoadedMetadata={handleVideoLoaded}
                  />
                  {selectedFrame ? (
                    <p className="player-note">
                      Jumped to frame {selectedFrame.frame} (
                      {(selectedFrame.frame / fps).toFixed(2)}
                      s) in cluster {selectedFrame.cluster}.
                    </p>
                  ) : (
                    <p className="player-note">Select a frame to seek within the clip.</p>
                  )}
                </>
              ) : (
                <div className="player-placeholder">
                  <p>No segment selected yet.</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}

export default App
