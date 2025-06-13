'use client'

import { useEffect, useRef, useState } from 'react'
import {
  ClickInteraction,
  DragNodeInteraction,
  ZoomInteraction,
  PanInteraction,
} from '@neo4j-nvl/interaction-handlers'
import { NVL, NvlOptions } from '@neo4j-nvl/base'
import {
  Node as NvlNode,
  Relationship as NvlRelationship,
} from '@neo4j-nvl/base'

interface Node {
  id: string
  caption: string
  color?: string
  properties?: Record<string, any>
  displayName?: string // Added for custom display name
}

interface Relationship {
  id: string
  from: string
  to: string
  caption: string
}

interface GraphVisualizationProps {
  nodes: Node[]
  relationships: Relationship[]
  onNodeClick?: (nodeData: Node) => void
}

export default function Neo4jNVLComponents({
  nodes,
  relationships,
  onNodeClick,
}: GraphVisualizationProps) {
  const graphContainerRef = useRef<HTMLDivElement>(null)
  const nvlInstanceRef = useRef<NVL | null>(null)

  // Keep a mapping between NVL node ids and our node objects with properties
  const nodeMapRef = useRef<Map<string, Node>>(new Map())

  // Process nodes to add display names based on node type
  const processedNodes = nodes.map((node) => {
    let displayName = node.caption

    // Set display name based on node type and available properties
    if (node.properties) {
      switch (node.caption) {
        case 'Function':
          displayName = node.properties.function_name || node.caption
          break
        case 'ApiEndpoint':
          displayName = node.properties.endpoint || node.caption
          break
        case 'DataFile':
        case 'DocumentationFile':
        case 'TemplateMarkupFile':
          displayName = node.properties.file_name || node.caption
          break
        case 'Folder':
          displayName = node.properties.folder_name || node.caption
          break
        default:
          // For other node types, try to find a name property or use caption
          displayName = node.properties.name || node.caption
      }
    }

    return {
      ...node,
      displayName,
    }
  })

  useEffect(() => {
    // Update the node map whenever nodes change
    nodeMapRef.current = new Map(processedNodes.map((node) => [node.id, node]))
  }, [processedNodes])

  useEffect(() => {
    if (
      !graphContainerRef.current ||
      processedNodes.length === 0 ||
      relationships.length === 0
    ) {
      return
    }

    // Map processed nodes to the format expected by NVL
    const nvlNodes = processedNodes.map((node) => ({
      id: node.id,
      caption: node.displayName || node.caption, // Use the display name instead of the type
      color: node.color,
    }))

    const options: NvlOptions = {
      layout: 'forceDirected',
    }

    const nvl = new NVL(
      graphContainerRef.current,
      nvlNodes,
      relationships,
      options
    )
    nvlInstanceRef.current = nvl

    // Set up interactions
    const clickInteraction = new ClickInteraction(nvl)
    const panInteraction = new PanInteraction(nvl)
    const dragNodeInteraction = new DragNodeInteraction(nvl)
    const zoomInteraction = new ZoomInteraction(nvl)

    clickInteraction.updateCallback('onNodeClick', (node: NvlNode) => {
      // Get the full node data from our map
      const nodeData = nodeMapRef.current.get(node.id)
      if (nodeData) {
        // Log node properties
        console.log('Node clicked:', nodeData)
        console.log('Node properties:', nodeData.properties || {})
        console.log('Display name:', nodeData.displayName)

        // Call the callback if provided
        if (onNodeClick) {
          onNodeClick(nodeData)
        }
      }
    })

    dragNodeInteraction.updateCallback('onDrag', (nodes) => {})
    zoomInteraction.updateCallback('onZoom', (zoomLevel) => {})
    panInteraction.updateCallback('onPan', (panning) => {})

    return () => {
      clickInteraction.destroy()
      dragNodeInteraction.destroy()
      zoomInteraction.destroy()
      panInteraction.destroy()
      nvl.destroy()
      nvlInstanceRef.current = null
    }
  }, [processedNodes, relationships, onNodeClick])

  return <div ref={graphContainerRef} className='w-full h-full' />
}
