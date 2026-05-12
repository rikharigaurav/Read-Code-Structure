'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import dynamic from 'next/dynamic'
import * as neo4j from 'neo4j-driver'
import type {
  Node as Neo4jNode,
  Relationship as Neo4jRelationship,
  Path as Neo4jPath,
} from 'neo4j-driver'
import {
  Play,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Zap,
  ZapOff,
  GitBranch,
  Layers,
  Search,
  X,
  ChevronDown,
  ChevronRight,
  Loader2,
  Database,
  Share2,
  NetworkIcon,
  FileCode2,
  BookOpen,
  Folder as FolderIcon,
  Braces,
  Globe,
  LayoutTemplate,
  FlaskConical,
  HelpCircle,
  Copy,
  Check,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import React from 'react'
import type { VisGraphCanvasHandle } from './vis-graph-canvas'

// ─── Shared types ─────────────────────────────────────────────────────────────
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

// Dynamically load vis-network canvas (no SSR).
// Cast restores the forwardRef typing that next/dynamic erases.
const VisGraphCanvas = dynamic(
  () => import('./vis-graph-canvas'),
  { ssr: false }
) as React.ForwardRefExoticComponent<
  VisGraphCanvasProps & React.RefAttributes<VisGraphCanvasHandle>
>

interface Neo4jGraphProps {
  uri: string
  user: string
  password: string
}

// ─── Node type config ─────────────────────────────────────────────────────────
const NODE_CONFIG: Record<
  string,
  { color: string; label: string; Icon: React.ComponentType<any> }
> = {
  DataFile:          { color: '#FF6B6B', label: 'Source File',  Icon: FileCode2 },
  DocumentationFile: { color: '#4D8AF0', label: 'Documentation',Icon: BookOpen },
  Folder:            { color: '#6BCB77', label: 'Folder',       Icon: FolderIcon },
  Function:          { color: '#845EC2', label: 'Function',     Icon: Braces },
  ApiEndpoint:       { color: '#F97316', label: 'API Endpoint', Icon: Globe },
  TemplateMarkupFile:{ color: '#F59E0B', label: 'Template',     Icon: LayoutTemplate },
  TestingFile:       { color: '#60A5FA', label: 'Test File',    Icon: FlaskConical },
}

const DEFAULT_COLOR = '#94A3B8'

// ─── Relationship colors ──────────────────────────────────────────────────────
const REL_COLORS: Record<string, string> = {
  BELONGS_TO: '#6BCB77',
  CALLS:      '#F97316',
  TEST:       '#60A5FA',
  GET:        '#22D3EE',
  POST:       '#A78BFA',
  PUT:        '#F59E0B',
  DELETE:     '#F87171',
  PATCH:      '#34D399',
}

// ─── Query templates ──────────────────────────────────────────────────────────
const QUERY_TEMPLATES = [
  { label: 'All Relationships (25)',    query: 'MATCH p=()-[]->() RETURN p LIMIT 25' },
  { label: 'File → Function calls',    query: 'MATCH p=(f:DataFile)-[:BELONGS_TO*0..1]-(fn:Function) RETURN p LIMIT 40' },
  { label: 'API Endpoints',            query: 'MATCH p=(f)-[r:GET|POST|PUT|DELETE|PATCH]->(api:ApiEndpoint) RETURN p LIMIT 30' },
  { label: 'Folder tree',              query: 'MATCH p=(:Folder)-[:BELONGS_TO*1..3]->(:Folder) RETURN p LIMIT 30' },
  { label: 'Test coverage',            query: 'MATCH p=(t:TestingFile)-[:TEST]->(n) RETURN p LIMIT 30' },
  { label: 'Docs & templates',         query: 'MATCH p=(n)-[]->() WHERE n:DocumentationFile OR n:TemplateMarkupFile RETURN p LIMIT 20' },
  { label: 'Nodes with most edges',    query: 'MATCH (n)-[r]->() RETURN n, count(r) AS c ORDER BY c DESC LIMIT 10' },
  { label: 'Count all node types',     query: 'MATCH (n) RETURN labels(n)[0] AS type, count(*) AS total ORDER BY total DESC' },
]

// ─── Helper: derive display name ──────────────────────────────────────────────
function getDisplayName(node: Neo4jNode, label: string): string {
  const p = node.properties
  switch (label) {
    case 'Function':           return p.function_name?.toString() || label
    case 'ApiEndpoint':        return p.endpoint_name?.toString() || p.endpoint?.toString() || label
    case 'Folder':             return p.folder_name?.toString() || label
    default:                   return p.file_name?.toString() || p.name?.toString() || label
  }
}

// ─── Component ────────────────────────────────────────────────────────────────
export function Neo4jGraph({ uri, user, password }: Neo4jGraphProps) {
  const [queryInput, setQueryInput] = useState(QUERY_TEMPLATES[0].query)
  const [nodes, setNodes] = useState<NodeData[]>([])
  const [relationships, setRelationships] = useState<EdgeData[]>([])
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isStabilizing, setIsStabilizing] = useState(false)
  const [isMounted, setIsMounted] = useState(false)
  const [selectedNode, setSelectedNode] = useState<NodeData | null>(null)
  const [physicsEnabled, setPhysicsEnabled] = useState(true)
  const [graphLayout, setGraphLayout] = useState<'forceDirected' | 'hierarchical'>('forceDirected')
  const [searchTerm, setSearchTerm] = useState('')
  const [copiedQuery, setCopiedQuery] = useState(false)

  const canvasRef = useRef<VisGraphCanvasHandle>(null)

  useEffect(() => { setIsMounted(true) }, [])

  // ── Derived counts ─────────────────────────────────────────────────────────
  const nodeTypeCounts = nodes.reduce<Record<string, number>>((acc, n) => {
    acc[n.caption] = (acc[n.caption] || 0) + 1
    return acc
  }, {})

  const relTypeCounts = relationships.reduce<Record<string, number>>((acc, r) => {
    acc[r.caption] = (acc[r.caption] || 0) + 1
    return acc
  }, {})

  const filteredNodeTypes = Object.entries(nodeTypeCounts).filter(([type]) =>
    !searchTerm || type.toLowerCase().includes(searchTerm.toLowerCase())
  )

  // ── Process raw Neo4j records ──────────────────────────────────────────────
  const processRecords = useCallback((records: neo4j.Record[]): { nodes: NodeData[]; relationships: EdgeData[] } => {
    const nodesMap = new Map<string, NodeData>()
    const relsMap = new Map<string, EdgeData>()

    records.forEach(record => {
      record.keys.forEach(key => {
        const value = record.get(key)

        if (value instanceof neo4j.types.Path) {
          const path = value as Neo4jPath
          path.segments.forEach(seg => {
            const addNode = (n: Neo4jNode) => {
              const lbl = n.labels[0] || 'Unknown'
              nodesMap.set(n.elementId, {
                id: n.elementId,
                caption: lbl,
                color: NODE_CONFIG[lbl]?.color || DEFAULT_COLOR,
                properties: n.properties,
                displayName: getDisplayName(n, lbl),
              })
            }
            addNode(seg.start)
            addNode(seg.end)
            const rel = seg.relationship as Neo4jRelationship
            relsMap.set(rel.elementId, {
              id: rel.elementId,
              from: seg.start.elementId,
              to: seg.end.elementId,
              caption: rel.type,
            })
          })
        } else if (value instanceof neo4j.types.Node) {
          const lbl = value.labels[0] || 'Unknown'
          nodesMap.set(value.elementId, {
            id: value.elementId,
            caption: lbl,
            color: NODE_CONFIG[lbl]?.color || DEFAULT_COLOR,
            properties: value.properties,
            displayName: getDisplayName(value, lbl),
          })
        }
      })
    })

    return { nodes: Array.from(nodesMap.values()), relationships: Array.from(relsMap.values()) }
  }, [])

  // ── Run Cypher ─────────────────────────────────────────────────────────────
  const handleRunQuery = async () => {
    setIsLoading(true)
    setError(null)
    setSelectedNode(null)
    setIsStabilizing(true)

    const driver = neo4j.driver(uri, neo4j.auth.basic(user, password))
    const session = driver.session()

    try {
      const result = await session.run(queryInput)
      const processed = processRecords(result.records)
      setNodes(processed.nodes)
      setRelationships(processed.relationships)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
      setIsStabilizing(false)
    } finally {
      setIsLoading(false)
      await session.close()
      await driver.close()
    }
  }

  const handleTemplateSelect = (label: string) => {
    const t = QUERY_TEMPLATES.find(t => t.label === label)
    if (t) setQueryInput(t.query)
  }

  const handlePhysicsToggle = () => {
    const next = !physicsEnabled
    setPhysicsEnabled(next)
    canvasRef.current?.togglePhysics(next)
  }

  const handleLayoutToggle = () => {
    setGraphLayout(prev => prev === 'forceDirected' ? 'hierarchical' : 'forceDirected')
  }

  const handleCopyQuery = async () => {
    await navigator.clipboard.writeText(queryInput)
    setCopiedQuery(true)
    setTimeout(() => setCopiedQuery(false), 1500)
  }

  // ─── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="w-full h-full flex flex-col bg-[#0a0f1e] text-slate-100 overflow-hidden rounded-lg border border-slate-800">

      {/* ── Top query bar ────────────────────────────────────────────────────── */}
      <div className="flex items-center gap-2 px-3 py-2 bg-[#0d1424] border-b border-slate-800 flex-shrink-0">
        <Database className="w-4 h-4 text-slate-500 flex-shrink-0" />

        <Select onValueChange={handleTemplateSelect}>
          <SelectTrigger className="w-52 h-8 bg-slate-800/60 border-slate-700 text-slate-300 text-xs">
            <SelectValue placeholder="Query templates…" />
          </SelectTrigger>
          <SelectContent className="bg-slate-900 border-slate-700">
            {QUERY_TEMPLATES.map(t => (
              <SelectItem key={t.label} value={t.label} className="text-slate-300 text-xs focus:bg-slate-800">
                {t.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <div className="flex-1 relative">
          <textarea
            value={queryInput}
            onChange={e => setQueryInput(e.target.value)}
            rows={1}
            onKeyDown={e => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), handleRunQuery())}
            className="w-full px-3 py-1.5 bg-slate-800/60 border border-slate-700 rounded-md text-slate-200 text-xs
              font-mono resize-none focus:outline-none focus:ring-1 focus:ring-blue-500/60 placeholder-slate-600
              leading-5"
            placeholder="Enter Cypher query… (Enter to run)"
            style={{ minHeight: 32, maxHeight: 64 }}
          />
        </div>

        <Button
          variant="ghost"
          size="sm"
          onClick={handleCopyQuery}
          className="h-8 w-8 p-0 text-slate-500 hover:text-slate-300"
          title="Copy query"
        >
          {copiedQuery ? <Check className="w-3.5 h-3.5 text-green-400" /> : <Copy className="w-3.5 h-3.5" />}
        </Button>

        <Button
          onClick={handleRunQuery}
          disabled={isLoading}
          size="sm"
          className="h-8 px-3 bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium flex-shrink-0 gap-1.5"
        >
          {isLoading
            ? <Loader2 className="w-3.5 h-3.5 animate-spin" />
            : <Play className="w-3.5 h-3.5 fill-current" />}
          {isLoading ? 'Running…' : 'Run'}
        </Button>
      </div>

      {/* ── Error banner ─────────────────────────────────────────────────────── */}
      {error && (
        <div className="flex items-center gap-2 px-4 py-2 bg-red-950/60 border-b border-red-800/50 text-red-300 text-xs flex-shrink-0">
          <span className="flex-1 font-mono">{error}</span>
          <button onClick={() => setError(null)}><X className="w-3.5 h-3.5" /></button>
        </div>
      )}

      {/* ── Main area ────────────────────────────────────────────────────────── */}
      <div className="flex flex-1 overflow-hidden">

        {/* Canvas area */}
        <div className="relative flex-1 overflow-hidden"
          style={{
            background: '#0a0f1e',
            backgroundImage: 'radial-gradient(circle, #1e293b 1px, transparent 1px)',
            backgroundSize: '24px 24px',
          }}
        >
          {/* Graph */}
          {nodes.length === 0 && !isLoading ? (
            <EmptyState onRun={handleRunQuery} />
          ) : isMounted ? (
            <VisGraphCanvas
              ref={canvasRef}
              nodes={nodes}
              relationships={relationships}
              onNodeClick={node => setSelectedNode(node)}
              onStabilized={() => setIsStabilizing(false)}
              physicsEnabled={physicsEnabled}
              layout={graphLayout}
            />
          ) : null}

          {/* Stabilizing badge */}
          {isStabilizing && nodes.length > 0 && (
            <div className="absolute top-3 left-1/2 -translate-x-1/2 bg-slate-900/90 text-slate-300
              text-xs px-3 py-1.5 rounded-full flex items-center gap-2 border border-slate-700 shadow-lg">
              <Loader2 className="w-3 h-3 animate-spin text-blue-400" />
              Arranging {nodes.length} nodes…
            </div>
          )}

          {/* Floating graph controls */}
          <div className="absolute bottom-4 left-4 flex flex-col gap-1">
            <ControlBtn onClick={() => canvasRef.current?.zoomIn()}    title="Zoom in"         Icon={ZoomIn} />
            <ControlBtn onClick={() => canvasRef.current?.zoomOut()}   title="Zoom out"        Icon={ZoomOut} />
            <ControlBtn onClick={() => canvasRef.current?.fitToScreen()} title="Fit to screen" Icon={Maximize2} />
            <div className="w-full h-px bg-slate-700 my-0.5" />
            <ControlBtn
              onClick={handlePhysicsToggle}
              title={physicsEnabled ? 'Disable physics' : 'Enable physics'}
              Icon={physicsEnabled ? Zap : ZapOff}
              active={physicsEnabled}
              activeColor="text-yellow-400"
            />
            <ControlBtn
              onClick={handleLayoutToggle}
              title={graphLayout === 'forceDirected' ? 'Switch to hierarchical' : 'Switch to force-directed'}
              Icon={graphLayout === 'forceDirected' ? GitBranch : Layers}
              active={graphLayout === 'hierarchical'}
              activeColor="text-purple-400"
            />
          </div>

          {/* Stats pill (bottom-right) */}
          {nodes.length > 0 && (
            <div className="absolute bottom-4 right-4 flex items-center gap-3 bg-slate-900/90 border
              border-slate-700 rounded-full px-4 py-1.5 text-xs text-slate-400 shadow-lg">
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full bg-blue-400 inline-block" />
                {nodes.length} nodes
              </span>
              <span className="text-slate-700">·</span>
              <span className="flex items-center gap-1.5">
                <Share2 className="w-3 h-3" />
                {relationships.length} edges
              </span>
            </div>
          )}
        </div>

        {/* ── Sidebar ──────────────────────────────────────────────────────── */}
        <aside className="w-72 flex-shrink-0 flex flex-col bg-[#0d1424] border-l border-slate-800 overflow-hidden">

          {/* Search */}
          <div className="px-3 pt-3 pb-2 border-b border-slate-800">
            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-slate-500" />
              <input
                value={searchTerm}
                onChange={e => setSearchTerm(e.target.value)}
                placeholder="Filter node types…"
                className="w-full pl-8 pr-8 py-1.5 bg-slate-800/60 border border-slate-700 rounded-md
                  text-xs text-slate-300 placeholder-slate-600 focus:outline-none focus:ring-1 focus:ring-blue-500/60"
              />
              {searchTerm && (
                <button onClick={() => setSearchTerm('')}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300">
                  <X className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          </div>

          <div className="flex-1 overflow-y-auto scrollbar-thin scrollbar-thumb-slate-700 scrollbar-track-transparent">

            {/* Node type legend */}
            {filteredNodeTypes.length > 0 && (
              <SidebarSection title="Node Types" count={Object.keys(nodeTypeCounts).length} icon={<NetworkIcon className="w-3.5 h-3.5" />}>
                <div className="space-y-1">
                  {filteredNodeTypes.map(([type, count]) => {
                    const cfg = NODE_CONFIG[type]
                    const Icon = cfg?.Icon || HelpCircle
                    const color = cfg?.color || DEFAULT_COLOR
                    return (
                      <div key={type} className="flex items-center gap-2.5 px-1 py-1.5 rounded-md
                        hover:bg-slate-800/60 cursor-default group transition-colors">
                        <div className="w-6 h-6 rounded-full flex items-center justify-center flex-shrink-0"
                          style={{ backgroundColor: color + '22', border: `1.5px solid ${color}` }}>
                          <Icon className="w-3 h-3" style={{ color }} />
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="text-xs text-slate-300 truncate">{cfg?.label || type}</div>
                          <div className="text-[10px] text-slate-600 truncate font-mono">{type}</div>
                        </div>
                        <Badge variant="outline"
                          className="text-[10px] h-4 px-1.5 border-slate-700 text-slate-500 tabular-nums">
                          {count}
                        </Badge>
                      </div>
                    )
                  })}
                </div>
              </SidebarSection>
            )}

            {/* Relationship types */}
            {Object.keys(relTypeCounts).length > 0 && (
              <SidebarSection title="Relationships" count={relationships.length} icon={<Share2 className="w-3.5 h-3.5" />}>
                <div className="space-y-1">
                  {Object.entries(relTypeCounts).map(([type, count]) => {
                    const color = REL_COLORS[type] || '#94A3B8'
                    return (
                      <div key={type} className="flex items-center gap-2.5 px-1 py-1 rounded-md hover:bg-slate-800/60 transition-colors">
                        <div className="flex items-center gap-1 flex-shrink-0">
                          <div className="w-4 h-[1.5px]" style={{ backgroundColor: color }} />
                          <div className="w-0 h-0 border-l-[5px] border-y-[3px] border-y-transparent"
                            style={{ borderLeftColor: color }} />
                        </div>
                        <span className="flex-1 text-xs text-slate-400 font-mono truncate">{type}</span>
                        <Badge variant="outline"
                          className="text-[10px] h-4 px-1.5 border-slate-700 text-slate-500 tabular-nums">
                          {count}
                        </Badge>
                      </div>
                    )
                  })}
                </div>
              </SidebarSection>
            )}

            {/* Selected node */}
            <SidebarSection
              title={selectedNode ? (selectedNode.displayName || selectedNode.caption) : 'Node Inspector'}
              icon={selectedNode
                ? <div className="w-3.5 h-3.5 rounded-full flex-shrink-0"
                    style={{ backgroundColor: NODE_CONFIG[selectedNode.caption]?.color || DEFAULT_COLOR }} />
                : <HelpCircle className="w-3.5 h-3.5" />}
              defaultOpen
            >
              {selectedNode ? (
                <div className="space-y-2">
                  <div className="flex items-center gap-2">
                    {(() => {
                      const cfg = NODE_CONFIG[selectedNode.caption]
                      const Icon = cfg?.Icon || HelpCircle
                      const color = cfg?.color || DEFAULT_COLOR
                      return (
                        <div className="w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0"
                          style={{ backgroundColor: color + '22', border: `2px solid ${color}` }}>
                          <Icon className="w-4 h-4" style={{ color }} />
                        </div>
                      )
                    })()}
                    <div>
                      <div className="text-xs font-semibold text-slate-200 break-all leading-tight">
                        {selectedNode.displayName || selectedNode.caption}
                      </div>
                      <div className="text-[10px] text-slate-500 font-mono mt-0.5">{selectedNode.caption}</div>
                    </div>
                  </div>

                  {selectedNode.properties && Object.keys(selectedNode.properties).length > 0 && (
                    <div className="bg-slate-900/60 rounded-md border border-slate-800 overflow-hidden">
                      {Object.entries(selectedNode.properties).map(([k, v], i) => (
                        <div key={k}
                          className={`flex gap-2 px-2.5 py-1.5 text-xs ${i > 0 ? 'border-t border-slate-800/60' : ''}`}>
                          <span className="text-slate-500 font-mono flex-shrink-0 w-24 truncate">{k}</span>
                          <span className="text-slate-300 break-all font-mono text-[10px]">
                            {typeof v === 'object' ? JSON.stringify(v) : String(v).slice(0, 120)}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-xs text-slate-600 italic px-1">Click any node in the graph to inspect its properties.</p>
              )}
            </SidebarSection>

          </div>
        </aside>
      </div>
    </div>
  )
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function ControlBtn({
  onClick, title, Icon, active = false, activeColor = 'text-blue-400',
}: {
  onClick: () => void
  title: string
  Icon: React.ComponentType<any>
  active?: boolean
  activeColor?: string
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      className={`w-8 h-8 rounded-md flex items-center justify-center transition-colors
        bg-slate-900/90 border border-slate-700 shadow-lg
        hover:bg-slate-800 hover:border-slate-600
        ${active ? activeColor : 'text-slate-400 hover:text-slate-200'}`}
    >
      <Icon className="w-3.5 h-3.5" />
    </button>
  )
}

function SidebarSection({
  title,
  icon,
  count,
  defaultOpen = false,
  children,
}: {
  title: string
  icon?: React.ReactNode
  count?: number
  defaultOpen?: boolean
  children: React.ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="border-b border-slate-800/60">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center gap-2 px-3 py-2.5 text-left hover:bg-slate-800/40 transition-colors"
      >
        <span className="text-slate-500">{icon}</span>
        <span className="flex-1 text-xs font-semibold text-slate-400 uppercase tracking-wider truncate">{title}</span>
        {count !== undefined && (
          <span className="text-[10px] text-slate-600 tabular-nums mr-1">{count}</span>
        )}
        {open
          ? <ChevronDown className="w-3.5 h-3.5 text-slate-600 flex-shrink-0" />
          : <ChevronRight className="w-3.5 h-3.5 text-slate-600 flex-shrink-0" />}
      </button>
      {open && <div className="px-3 pb-3">{children}</div>}
    </div>
  )
}

function EmptyState({ onRun }: { onRun: () => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-full gap-5 px-8 text-center">
      <div className="relative">
        <div className="w-20 h-20 rounded-full bg-slate-800/60 border border-slate-700 flex items-center justify-center">
          <NetworkIcon className="w-9 h-9 text-slate-600" />
        </div>
        <div className="absolute -top-1 -right-1 w-6 h-6 rounded-full bg-blue-600/20 border border-blue-500/40
          flex items-center justify-center">
          <Database className="w-3 h-3 text-blue-400" />
        </div>
      </div>
      <div>
        <p className="text-slate-300 font-semibold mb-1">No graph data yet</p>
        <p className="text-slate-600 text-xs leading-relaxed">
          Select a query template or write a Cypher query,<br />then click <strong className="text-slate-400">Run</strong> to visualize the codebase graph.
        </p>
      </div>
      <button
        onClick={onRun}
        className="flex items-center gap-2 px-4 py-2 bg-blue-600/20 hover:bg-blue-600/30 border border-blue-500/40
          rounded-md text-blue-300 text-xs font-medium transition-colors"
      >
        <Play className="w-3.5 h-3.5 fill-current" />
        Run default query
      </button>
    </div>
  )
}
