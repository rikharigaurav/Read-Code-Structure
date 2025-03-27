'use client'

import { useState, useEffect } from 'react'
import { ChevronRight, ChevronDown, File, Folder, Loader2 } from 'lucide-react'
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

export function FileTree({ rootPath, onSelect }: FileTreeProps) {
  const [data, setData] = useState<TreeNode[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchDirectory = async (path: string) => {
    try {
      const response = await axios.get(
        `http://localhost:8000/directory?path=${encodeURIComponent(
          path
        )}`
      )

      return response.data
    } catch (err) {
      let errorMessage = 'Failed to fetch directory contents'
      if (axios.isAxiosError(err)) {
        errorMessage = err.response?.data?.error || err.message
      }
      throw new Error(errorMessage)
    }
  }

  useEffect(() => {
    const loadStructure = async () => {
      try {
        setLoading(true)
        setError(null)
        console.log('Loading structure for path:', rootPath)

        const initialData = await fetchDirectory(rootPath)
        console.log('Received initial data:', initialData)

        setData(initialData)
      } catch (err) {
        console.error('Error loading structure:', err)
        setError(
          err instanceof Error ? err.message : 'Failed to load structure'
        )
      } finally {
        setLoading(false)
      }
    }

    if (rootPath) {
      loadStructure()
    } else {
      setError('No root path provided')
      setLoading(false)
    }
  }, [rootPath])

  const TreeNode = ({
    node,
    level = 0,
  }: {
    node: TreeNode
    level?: number
  }) => {
    const [isOpen, setIsOpen] = useState(false)
    const [children, setChildren] = useState<TreeNode[]>([])
    const [childrenLoading, setChildrenLoading] = useState(false)

    const loadChildren = async () => {
      if (node.type === 'directory' && !children.length) {
        try {
          setChildrenLoading(true)
          console.log('Loading children for:', node.path)

          const childNodes = await fetchDirectory(node.path)
          console.log('Received children:', childNodes)

          setChildren(childNodes)
        } catch (err) {
          console.error('Error loading children:', err)
        } finally {
          setChildrenLoading(false)
        }
      }
    }

    return (
      <div>
        <div
          className={cn(
            'flex items-center gap-2 p-1 rounded-md hover:bg-accent cursor-pointer',
            'text-sm text-muted-foreground hover:text-foreground'
          )}
          style={{ paddingLeft: `${level * 12}px` }}
          onClick={async () => {
            if (node.type === 'directory') {
              await loadChildren()
              setIsOpen(!isOpen)
            } else {
              onSelect(node.path)
            }
          }}
        >
          {node.type === 'directory' ? (
            <>
              {childrenLoading ? (
                <Loader2 className='w-4 h-4 animate-spin' />
              ) : isOpen ? (
                <ChevronDown className='w-4 h-4' />
              ) : (
                <ChevronRight className='w-4 h-4' />
              )}
              <Folder className='w-4 h-4' />
            </>
          ) : (
            <>
              <span className='w-4' />
              <File className='w-4 h-4' />
            </>
          )}
          <span className='truncate'>{node.name}</span>
        </div>
        {isOpen && node.type === 'directory' && (
          <div>
            {children.map((child, index) => (
              <TreeNode
                key={`${child.path}-${index}`}
                node={child}
                level={level + 1}
              />
            ))}
          </div>
        )}
      </div>
    )
  }

  if (loading) {
    return (
      <div className='flex items-center justify-center h-full'>
        <Loader2 className='w-8 h-8 animate-spin text-muted-foreground' />
      </div>
    )
  }

  if (error) {
    return (
      <div className='p-4 text-red-500 text-sm'>
        Error: {error}
        <div className='mt-2 text-xs'>
          Check that:
          <ul className='list-disc pl-4'>
            <li>{rootPath}</li>
            <li>Backend server is running on port 8000</li>
            <li>CORS is properly configured</li>
            <li>Path is accessible to the server</li>
          </ul>
        </div>
      </div>
    )
  }

  return (
    <div className='space-y-1 overflow-auto'>
      {data.map((node, index) => (
        <TreeNode key={`${node.path}-${index}`} node={node} />
      ))}
    </div>
  )
}
