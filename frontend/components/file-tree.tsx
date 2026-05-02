'use client'

import { useState, useEffect } from 'react'
import {
  ChevronRight,
  ChevronDown,
  Loader2,
  FileCode2,
  FileText,
  FileJson,
  FileImage,
  Folder,
  FolderOpen,
  Settings,
  TestTube2,
  Globe,
  Braces,
  BookOpen,
  FileType2,
  Terminal,
  Database,
  Package,
  Lock,
  AlertCircle,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import axios from 'axios'

interface TreeNode {
  name: string
  type: 'file' | 'directory'
  path: string
  children?: TreeNode[]
}

interface FileTreeProps {
  rootPath: string
  onSelect: (filePath: string) => void
}

// ─── Extension → icon + color map ────────────────────────────────────────────
type FileIconConfig = {
  Icon: React.ComponentType<any>
  color: string
}

const EXT_MAP: Record<string, FileIconConfig> = {
  // Python
  py:   { Icon: FileCode2,  color: '#3B82F6' },
  pyi:  { Icon: FileCode2,  color: '#60A5FA' },
  // JavaScript / TypeScript
  js:   { Icon: FileCode2,  color: '#EAB308' },
  jsx:  { Icon: FileCode2,  color: '#06B6D4' },
  ts:   { Icon: FileCode2,  color: '#3B82F6' },
  tsx:  { Icon: FileCode2,  color: '#06B6D4' },
  mjs:  { Icon: FileCode2,  color: '#EAB308' },
  cjs:  { Icon: FileCode2,  color: '#EAB308' },
  // Web
  html: { Icon: Globe,      color: '#F97316' },
  htm:  { Icon: Globe,      color: '#F97316' },
  css:  { Icon: FileType2,  color: '#EC4899' },
  scss: { Icon: FileType2,  color: '#EC4899' },
  sass: { Icon: FileType2,  color: '#EC4899' },
  less: { Icon: FileType2,  color: '#A78BFA' },
  // Data / Config
  json: { Icon: Braces,     color: '#F59E0B' },
  yaml: { Icon: Settings,   color: '#8B5CF6' },
  yml:  { Icon: Settings,   color: '#8B5CF6' },
  toml: { Icon: Settings,   color: '#F97316' },
  ini:  { Icon: Settings,   color: '#94A3B8' },
  cfg:  { Icon: Settings,   color: '#94A3B8' },
  env:  { Icon: Lock,       color: '#F59E0B' },
  // Docs
  md:   { Icon: BookOpen,   color: '#94A3B8' },
  mdx:  { Icon: BookOpen,   color: '#60A5FA' },
  txt:  { Icon: FileText,   color: '#94A3B8' },
  rst:  { Icon: FileText,   color: '#94A3B8' },
  // Database
  sql:  { Icon: Database,   color: '#22D3EE' },
  db:   { Icon: Database,   color: '#22D3EE' },
  // Test files (by name convention handled separately)
  // Shell
  sh:   { Icon: Terminal,   color: '#34D399' },
  bash: { Icon: Terminal,   color: '#34D399' },
  zsh:  { Icon: Terminal,   color: '#34D399' },
  // Images
  png:  { Icon: FileImage,  color: '#A78BFA' },
  jpg:  { Icon: FileImage,  color: '#A78BFA' },
  jpeg: { Icon: FileImage,  color: '#A78BFA' },
  svg:  { Icon: FileImage,  color: '#F97316' },
  gif:  { Icon: FileImage,  color: '#A78BFA' },
  webp: { Icon: FileImage,  color: '#A78BFA' },
  // JSON variants
  jsonc:{ Icon: Braces,     color: '#F59E0B' },
  // Lock / manifest
  lock: { Icon: Lock,       color: '#94A3B8' },
  // Java / JVM
  java: { Icon: FileCode2,  color: '#F97316' },
  kt:   { Icon: FileCode2,  color: '#A78BFA' },
  // Go
  go:   { Icon: FileCode2,  color: '#06B6D4' },
  // Rust
  rs:   { Icon: FileCode2,  color: '#F97316' },
  // C / C++
  c:    { Icon: FileCode2,  color: '#60A5FA' },
  cpp:  { Icon: FileCode2,  color: '#60A5FA' },
  h:    { Icon: FileCode2,  color: '#94A3B8' },
  // Ruby
  rb:   { Icon: FileCode2,  color: '#EF4444' },
  // PHP
  php:  { Icon: FileCode2,  color: '#A78BFA' },
  // Package
  xml:  { Icon: FileJson,   color: '#F59E0B' },
}

function getFileIconConfig(name: string): FileIconConfig {
  // Test file heuristic
  if (name.match(/\.(test|spec)\.[jt]sx?$/) || name.startsWith('test_') || name.endsWith('_test.py')) {
    return { Icon: TestTube2, color: '#60A5FA' }
  }
  // Package / manifest
  if (name === 'package.json' || name === 'pyproject.toml' || name === 'setup.py' || name === 'Cargo.toml') {
    return { Icon: Package, color: '#F59E0B' }
  }
  // Dotfiles
  if (name === '.gitignore' || name === '.gitattributes') {
    return { Icon: Settings, color: '#94A3B8' }
  }
  if (name.startsWith('.env')) {
    return { Icon: Lock, color: '#F59E0B' }
  }
  if (name === 'Dockerfile' || name === 'docker-compose.yml' || name === 'docker-compose.yaml') {
    return { Icon: Package, color: '#06B6D4' }
  }
  if (name === 'Makefile') {
    return { Icon: Terminal, color: '#34D399' }
  }
  // By extension
  const ext = name.split('.').pop()?.toLowerCase() || ''
  return EXT_MAP[ext] || { Icon: FileText, color: '#64748B' }
}

// ─── Tree node component ──────────────────────────────────────────────────────
function TreeNodeItem({
  node,
  level = 0,
  onSelect,
  selectedPath,
}: {
  node: TreeNode
  level?: number
  onSelect: (path: string) => void
  selectedPath?: string | null
}) {
  const [isOpen, setIsOpen] = useState(false)
  const [children, setChildren] = useState<TreeNode[]>([])
  const [childrenLoading, setChildrenLoading] = useState(false)

  const isSelected = node.type === 'file' && node.path === selectedPath
  const { Icon, color } = getFileIconConfig(node.name)

  const loadChildren = async () => {
    if (node.type === 'directory' && !children.length) {
      try {
        setChildrenLoading(true)
        const res = await axios.get(`/api/directory?path=${encodeURIComponent(node.path)}`)
        setChildren(res.data)
      } catch {
        // silently fail
      } finally {
        setChildrenLoading(false)
      }
    }
  }

  const handleClick = async () => {
    if (node.type === 'directory') {
      await loadChildren()
      setIsOpen(o => !o)
    } else {
      onSelect(node.path)
    }
  }

  return (
    <div>
      <div
        onClick={handleClick}
        className={cn(
          'flex items-center gap-1.5 py-[3px] pr-2 rounded-md cursor-pointer select-none transition-colors group',
          isSelected
            ? 'bg-accent text-accent-foreground'
            : 'text-muted-foreground hover:text-foreground hover:bg-accent/50'
        )}
        style={{ paddingLeft: `${8 + level * 14}px` }}
      >
        {/* Expand/collapse for directories */}
        <span className="w-4 flex-shrink-0 flex items-center justify-center">
          {node.type === 'directory'
            ? childrenLoading
              ? <Loader2 className="w-3 h-3 animate-spin" />
              : isOpen
                ? <ChevronDown className="w-3.5 h-3.5 text-muted-foreground/70" />
                : <ChevronRight className="w-3.5 h-3.5 text-muted-foreground/50 group-hover:text-muted-foreground/70" />
            : null}
        </span>

        {/* Icon */}
        {node.type === 'directory' ? (
          isOpen
            ? <FolderOpen className="w-4 h-4 flex-shrink-0 text-yellow-400/80" />
            : <Folder className="w-4 h-4 flex-shrink-0 text-yellow-400/60 group-hover:text-yellow-400/80" />
        ) : (
          <Icon className="w-4 h-4 flex-shrink-0" style={{ color }} />
        )}

        {/* Name */}
        <span className={cn(
          'text-xs truncate flex-1',
          node.type === 'directory' ? 'font-medium' : '',
          isSelected ? 'text-accent-foreground font-medium' : ''
        )}>
          {node.name}
        </span>
      </div>

      {/* Children */}
      {isOpen && node.type === 'directory' && children.length > 0 && (
        <div className="relative">
          {/* Vertical indent guide line */}
          <div
            className="absolute top-0 bottom-0 w-px bg-border/50"
            style={{ left: `${8 + level * 14 + 9}px` }}
          />
          {children.map((child, i) => (
            <TreeNodeItem
              key={`${child.path}-${i}`}
              node={child}
              level={level + 1}
              onSelect={onSelect}
              selectedPath={selectedPath}
            />
          ))}
        </div>
      )}
    </div>
  )
}

// ─── Main FileTree ─────────────────────────────────────────────────────────────
export function FileTree({ rootPath, onSelect }: FileTreeProps) {
  const [data, setData] = useState<TreeNode[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedPath, setSelectedPath] = useState<string | null>(null)

  useEffect(() => {
    const load = async () => {
      if (!rootPath) {
        setError('No root path provided')
        setLoading(false)
        return
      }
      try {
        setLoading(true)
        setError(null)
        const res = await axios.get(`/api/directory?path=${encodeURIComponent(rootPath)}`)
        setData(res.data)
      } catch (err) {
        setError(axios.isAxiosError(err) ? err.response?.data?.error || err.message : 'Failed to load structure')
      } finally {
        setLoading(false)
      }
    }
    load()
  }, [rootPath])

  const handleSelect = (path: string) => {
    setSelectedPath(path)
    onSelect(path)
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-10">
        <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-3">
        <div className="flex items-start gap-2 text-destructive text-xs p-2 rounded-md bg-destructive/10 border border-destructive/20">
          <AlertCircle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
          <div>
            <p className="font-medium mb-1">Could not load files</p>
            <p className="text-muted-foreground">{error}</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-0.5">
      {data.map((node, i) => (
        <TreeNodeItem
          key={`${node.path}-${i}`}
          node={node}
          level={0}
          onSelect={handleSelect}
          selectedPath={selectedPath}
        />
      ))}
    </div>
  )
}
