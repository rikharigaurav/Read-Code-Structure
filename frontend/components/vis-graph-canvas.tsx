'use client'

import { useEffect, useRef, forwardRef, useImperativeHandle } from 'react'

interface NodeData {
  id: string
  caption: string
  color?: string
  properties?: Record<string, any>
  displayName?: string
}

interface EdgeData {
  id: string
  from: string
  to: string
  caption: string
}

interface VisGraphCanvasProps {
  nodes: NodeData[]
  relationships: EdgeData[]
  onNodeClick?: (node: NodeData | null) => void
  onStabilized?: () => void
  physicsEnabled?: boolean
  layout?: 'forceDirected' | 'hierarchical'
}

export interface VisGraphCanvasHandle {
  zoomIn(): void
  zoomOut(): void
  fitToScreen(): void
  togglePhysics(enabled: boolean): void
}

// ─── Icon factory ─────────────────────────────────────────────────────────────
// Each icon is a white Lucide-style path drawn inside a colored circle (64×64 SVG).
function makeSvgIcon(color: string, pathElements: string): string {
  const svg = [
    '<svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 64 64">',
    `<circle cx="32" cy="32" r="30" fill="${color}"/>`,
    // scale the 24-px Lucide viewBox up to fill the circle
    '<g transform="translate(8,8) scale(2)" stroke="white" fill="none"',
    ' stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">',
    pathElements,
    '</g>',
    '</svg>',
  ].join('')
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`
}

const ICONS: Record<string, string> = {
  DataFile: makeSvgIcon('#FF6B6B', [
    '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>',
    '<polyline points="14 2 14 8 20 8"/>',
    '<polyline points="10 13 8 15 10 17"/>',
    '<polyline points="14 13 16 15 14 17"/>',
  ].join('')),

  DocumentationFile: makeSvgIcon('#4D8AF0', [
    '<path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>',
    '<path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>',
  ].join('')),

  Folder: makeSvgIcon('#6BCB77', [
    '<path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>',
  ].join('')),

  Function: makeSvgIcon('#845EC2', [
    '<path d="M8 3H7a2 2 0 0 0-2 2v5a2 2 0 0 1-2 2 2 2 0 0 1 2 2v5c0 1.1.9 2 2 2h1"/>',
    '<path d="M16 21h1a2 2 0 0 0 2-2v-5c0-1.1.9-2 2-2a2 2 0 0 1-2-2V5a2 2 0 0 0-2-2h-1"/>',
  ].join('')),

  ApiEndpoint: makeSvgIcon('#F97316', [
    '<circle cx="12" cy="12" r="10"/>',
    '<path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>',
    '<line x1="2" y1="12" x2="22" y2="12"/>',
  ].join('')),

  TemplateMarkupFile: makeSvgIcon('#F59E0B', [
    '<rect x="3" y="3" width="18" height="18" rx="2"/>',
    '<line x1="3" y1="9" x2="21" y2="9"/>',
    '<line x1="9" y1="21" x2="9" y2="9"/>',
  ].join('')),

  TestingFile: makeSvgIcon('#60A5FA', [
    '<path d="M14 2v6l3 11H7L10 8V2"/>',
    '<line x1="8" y1="2" x2="16" y2="2"/>',
    '<path d="M6.4 15.1a9 9 0 0 0 11.2 0"/>',
  ].join('')),
}

const COLORS: Record<string, string> = {
  DataFile: '#FF6B6B',
  DocumentationFile: '#4D8AF0',
  Folder: '#6BCB77',
  Function: '#845EC2',
  ApiEndpoint: '#F97316',
  TemplateMarkupFile: '#F59E0B',
  TestingFile: '#60A5FA',
}

const DEFAULT_ICON = makeSvgIcon('#94A3B8', [
  '<circle cx="12" cy="12" r="10"/>',
  '<line x1="12" y1="8" x2="12" y2="16"/>',
  '<line x1="8" y1="12" x2="16" y2="12"/>',
].join(''))

// ─── Helpers ──────────────────────────────────────────────────────────────────
function clip(str: string, max = 22): string {
  return str.length <= max ? str : str.slice(0, max - 1) + '…'
}

function nodeTooltip(node: NodeData): string {
  const rows = Object.entries(node.properties || {})
    .slice(0, 6)
    .map(([k, v]) => {
      const val = typeof v === 'object' ? JSON.stringify(v) : String(v)
      return `<tr>
        <td style="color:#94a3b8;padding:2px 10px 2px 0;white-space:nowrap">${k}</td>
        <td style="color:#e2e8f0;word-break:break-all">${val.slice(0, 60)}</td>
      </tr>`
    })
    .join('')
  return `<div style="background:#1e293b;border:1px solid #334155;border-radius:8px;
    padding:10px 14px;max-width:320px;font-family:system-ui,sans-serif;font-size:12px;line-height:1.5">
    <div style="font-weight:700;color:#f1f5f9;margin-bottom:4px;font-size:13px">
      ${node.displayName || node.caption}
    </div>
    <div style="color:#64748b;margin-bottom:6px">${node.caption}</div>
    ${rows ? `<table style="border-collapse:collapse">${rows}</table>` : ''}
  </div>`
}

// ─── Component ────────────────────────────────────────────────────────────────
const VisGraphCanvas = forwardRef<VisGraphCanvasHandle, VisGraphCanvasProps>(
  function VisGraphCanvas(
    {
      nodes,
      relationships,
      onNodeClick,
      onStabilized,
      physicsEnabled = true,
      layout = 'forceDirected',
    },
    ref
  ) {
    const containerRef = useRef<HTMLDivElement>(null)
    const networkRef = useRef<any>(null)
    const nodeMapRef = useRef<Map<string, NodeData>>(new Map())

    useImperativeHandle(ref, () => ({
      zoomIn() {
        if (!networkRef.current) return
        const s = networkRef.current.getScale()
        networkRef.current.moveTo({ scale: s * 1.25, animation: { duration: 300, easingFunction: 'easeInOutQuad' } })
      },
      zoomOut() {
        if (!networkRef.current) return
        const s = networkRef.current.getScale()
        networkRef.current.moveTo({ scale: s * 0.8, animation: { duration: 300, easingFunction: 'easeInOutQuad' } })
      },
      fitToScreen() {
        networkRef.current?.fit({ animation: { duration: 600, easingFunction: 'easeInOutQuad' } })
      },
      togglePhysics(enabled: boolean) {
        networkRef.current?.setOptions({ physics: { enabled } })
      },
    }))

    // Keep a fresh node map for click lookups
    useEffect(() => {
      nodeMapRef.current = new Map(nodes.map(n => [n.id, n]))
    }, [nodes])

    useEffect(() => {
      if (!containerRef.current) return

      let mounted = true

      const init = async () => {
        const [{ Network }, { DataSet }] = await Promise.all([
          import('vis-network'),
          import('vis-data'),
        ])
        if (!mounted || !containerRef.current) return

        // ── Build vis node objects ──────────────────────────────────────────
        const visNodes = nodes.map(node => {
          const color = COLORS[node.caption] ?? '#94A3B8'
          return {
            id: node.id,
            label: clip(node.displayName || node.caption),
            shape: 'circularImage',
            image: ICONS[node.caption] ?? DEFAULT_ICON,
            size: 28,
            borderWidth: 2.5,
            borderWidthSelected: 5,
            color: {
              border: color,
              background: color + '22',
              highlight: { border: color, background: color + '44' },
              hover: { border: color, background: color + '33' },
            },
            font: {
              color: '#f1f5f9',
              size: 11,
              face: 'Inter, system-ui, sans-serif',
              strokeWidth: 3,
              strokeColor: '#0f172a',
            },
            shadow: { enabled: true, color: color + '55', size: 10, x: 0, y: 3 },
            title: nodeTooltip(node),
          }
        })

        // ── Build vis edge objects ──────────────────────────────────────────
        const visEdges = relationships.map(rel => ({
          id: rel.id,
          from: rel.from,
          to: rel.to,
          label: rel.caption,
          arrows: { to: { enabled: true, scaleFactor: 0.65, type: 'arrow' } },
          color: { color: '#334155', highlight: '#94a3b8', hover: '#475569' },
          font: {
            color: '#94a3b8',
            size: 10,
            face: 'monospace',
            strokeWidth: 2,
            strokeColor: '#0f172a',
            align: 'middle',
          },
          smooth: { enabled: true, type: 'dynamic' },
          width: 1.5,
          selectionWidth: 3,
          hoverWidth: 2.5,
        }))

        // ── Options ────────────────────────────────────────────────────────
        const options: any = {
          physics: {
            enabled: physicsEnabled,
            solver: 'forceAtlas2Based',
            forceAtlas2Based: {
              gravitationalConstant: -80,
              centralGravity: 0.01,
              springLength: 130,
              springConstant: 0.05,
              damping: 0.4,
              avoidOverlap: 0.6,
            },
            stabilization: { enabled: true, iterations: 300, updateInterval: 25 },
          },
          layout:
            layout === 'hierarchical'
              ? {
                  hierarchical: {
                    enabled: true,
                    direction: 'UD',
                    sortMethod: 'directed',
                    nodeSpacing: 120,
                    levelSeparation: 180,
                  },
                }
              : { improvedLayout: true, randomSeed: 42 },
          interaction: {
            hover: true,
            tooltipDelay: 200,
            zoomView: true,
            dragView: true,
            multiselect: false,
            navigationButtons: false,
            keyboard: false,
          },
        }

        const network = new Network(
          containerRef.current!,
          { nodes: new DataSet(visNodes), edges: new DataSet(visEdges) },
          options
        )
        networkRef.current = network

        network.on('click', (params: any) => {
          if (params.nodes.length > 0) {
            const data = nodeMapRef.current.get(params.nodes[0] as string)
            if (data) onNodeClick?.(data)
          } else {
            onNodeClick?.(null)
          }
        })

        network.on('stabilized', () => onStabilized?.())
      }

      init()

      return () => {
        mounted = false
        if (networkRef.current) {
          networkRef.current.destroy()
          networkRef.current = null
        }
      }
    }, [nodes, relationships, physicsEnabled, layout])

    return <div ref={containerRef} className="w-full h-full" />
  }
)

export default VisGraphCanvas
