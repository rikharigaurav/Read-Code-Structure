'use client'

import { useState, useEffect, useRef } from 'react'
import {
  ClickInteraction,
  DragNodeInteraction,
  ZoomInteraction,
  PanInteraction,
} from '@neo4j-nvl/interaction-handlers'
import * as neo4j from 'neo4j-driver'
import type {
  Node as Neo4jNode,
  Relationship as Neo4jRelationship,
  Path as Neo4jPath,
} from 'neo4j-driver'
import { Play } from 'lucide-react'
import { NVL, NvlOptions } from '@neo4j-nvl/base'

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { ChevronUp, ChevronDown } from 'lucide-react'
import { Node as node, Relationship as relstionship } from '@neo4j-nvl/base'

interface Node {
  id: string
  caption: string
  color?: string
}

interface Relationship {
  id: string
  from: string
  to: string
  caption: string
}

interface Neo4jGraphProps {
  uri: string
  user: string
  password: string
}

// Predefined query templates
const QUERY_TEMPLATES = [
  {
    label: 'Get All Relationships',
    query: 'MATCH p=()-[]->() RETURN p LIMIT 25;',
  },
  {
    label: 'Count Node Types',
    query: 'MATCH (n) RETURN labels(n) as NodeType, count(*) as Count;',
  },
  {
    label: 'Find Connected Nodes',
    query: 'MATCH (a)-[r]->(b) RETURN a.name, type(r), b.name LIMIT 25;',
  },
  {
    label: 'Shortest Paths',
    query: 'MATCH p=shortestPath((a)-[*]-(b)) RETURN p LIMIT 5;',
  },
  {
    label: 'Nodes with Most Connections',
    query:
      'MATCH (n)-[r]->() RETURN n, count(r) as connections ORDER BY connections DESC LIMIT 10;',
  },
]

export const Neo4jGraph = ({ uri, user, password }: Neo4jGraphProps) => {
  const [queryInput, setQueryInput] = useState(
    'MATCH p=()-[]->() RETURN p LIMIT 25;'
  )
  const [nodes, setNodes] = useState<Node[]>([])
  const [relationships, setRelationships] = useState<Relationship[]>([])
  const [error, setError] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [isNodesOpen, setIsNodesOpen] = useState(true)
  const [isRelationshipsOpen, setIsRelationshipsOpen] = useState(true)
  const graphContainerRef = useRef<HTMLDivElement>(null)
  const [nvlInstance, setNvlInstance] = useState<NVL | null>(null)

  const getColorForLabel = (label: string): string => {
    const colorMap: { [key: string]: string } = {
      DataFile: '#FF6B6B',
      DocumentationFile: '#4D8AF0',
      Folder: '#6BCB77',
      Function: '#845EC2',
      ApiEndpoint: '#FF8066',
      TemplateMarkupFile: '#FFB88C',
      TestingFile: '#A5A5A5',
    }
    return colorMap[label] || '#CCCCCC'
  }

  const processNeo4jData = (records: neo4j.Record[]) => {
    const nodesMap = new Map<string, Node>()
    const relationshipsMap = new Map<string, Relationship>()

    records.forEach((record: neo4j.Record) => {
      const path = record.get('p') as Neo4jPath

      path.segments.forEach(
        (segment: {
          start: Neo4jNode
          end: Neo4jNode
          relationship: Neo4jRelationship
        }) => {
          // Process start node
          const startNode = segment.start
          const startLabel = startNode.labels[0] || 'Unknown'
          nodesMap.set(startNode.elementId, {
            id: startNode.elementId,
            caption: startLabel,
            color: getColorForLabel(startLabel),
          })

          // Process relationship
          const relationship = segment.relationship
          relationshipsMap.set(relationship.elementId, {
            id: relationship.elementId,
            from: startNode.elementId,
            to: segment.end.elementId,
            caption: relationship.type,
          })

          // Process end node
          const endNode = segment.end
          const endLabel = endNode.labels[0] || 'Unknown'
          nodesMap.set(endNode.elementId, {
            id: endNode.elementId,
            caption: endLabel,
            color: getColorForLabel(endLabel),
          })
        }
      )
    })

    return {
      nodes: Array.from(nodesMap.values()),
      relationships: Array.from(relationshipsMap.values()),
    }
  }

  const handleRunQuery = async () => {
    setIsLoading(true)
    setError(null)
    const driver = neo4j.driver(uri, neo4j.auth.basic(user, password))
    const session = driver.session()

    try {
      const result = await session.run(queryInput)
      const { nodes, relationships } = processNeo4jData(result.records)

      console.log('Processed Nodes:')
      console.log(JSON.stringify(nodes, null, 2))
      console.log('Processed Relationships:')
      console.log(JSON.stringify(relationships, null, 2))

      setNodes(nodes)
      setRelationships(relationships)
    } catch (error) {
      setError(
        error instanceof Error ? error.message : 'An unknown error occurred'
      )
      console.error('Query Error:', error)
    } finally {
      setIsLoading(false)
      await session.close()
      await driver.close()
    }
  }

  // Handle selecting a predefined query template
  const handleQueryTemplateSelect = (value: string) => {
    const selectedTemplate = QUERY_TEMPLATES.find(
      (template) => template.label === value
    )
    if (selectedTemplate) {
      setQueryInput(selectedTemplate.query)
    }
  }

  // Get unique node types for the legend
  const nodeTypes = Array.from(new Set(nodes.map((node) => node.caption)))

  // Get unique relationship types for the legend
  const relationshipTypes = Array.from(
    new Set(relationships.map((rel) => rel.caption))
  )

  useEffect(() => {
    if (
      graphContainerRef.current &&
      nodes.length > 0 &&
      relationships.length > 0
    ) {
      const options: NvlOptions = {
        layout: 'forceDirected',
      }

      const nvl = new NVL(
        graphContainerRef.current,
        nodes,
        relationships,
        options
      )
      setNvlInstance(nvl)

      return () => {
        nvl.destroy() 
        setNvlInstance(null)
      }
    }
  }, [nodes, relationships])

  useEffect(() => {

    if (!nvlInstance) return
    const clickInteraction = new ClickInteraction(nvlInstance)
    const panInteraction = new PanInteraction(nvlInstance)
    const dragNodeInteraction = new DragNodeInteraction(nvlInstance)
    const zoomInteraction = new ZoomInteraction(nvlInstance)

    clickInteraction.updateCallback('onNodeClick', (node: node) => {})
    dragNodeInteraction.updateCallback('onDrag', (nodes) => {})
    zoomInteraction.updateCallback('onZoom', (zoomLevel) => {})
    panInteraction.updateCallback('onPan', (panning) => {})

    return () => {
      clickInteraction.destroy()
      dragNodeInteraction.destroy()
      zoomInteraction.destroy()
      panInteraction.destroy()
    }
  }, [nvlInstance])

  return (
    <div className='w-full h-screen flex flex-col bg-gray-900 border-rounded-t-md'>
      <div className='p-4 bg-gray-800 border-b border-gray-700 flex items-center gap-3 '>
        <Select onValueChange={handleQueryTemplateSelect}>
          <SelectTrigger className='w-[280px] bg-gray-700 text-gray-100'>
            <SelectValue placeholder='Select Query Template' />
          </SelectTrigger>
          <SelectContent>
            {QUERY_TEMPLATES.map((template, index) => (
              <SelectItem key={index} value={template.label}>
                {template.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <textarea
          value={queryInput}
          onChange={(e) => setQueryInput(e.target.value)}
          rows={2}
          className='flex-1 p-2 bg-gray-700 text-gray-100 border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 font-mono text-sm'
          placeholder='Enter Cypher query...'
        />
        <Button
          onClick={handleRunQuery}
          disabled={isLoading}
          className='px-3 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 flex items-center justify-center'
        >
          {isLoading ? (
            <div className='h-4 w-4 border-2 border-white border-t-transparent rounded-full animate-spin'></div>
          ) : (
            <Play className='h-4 w-4' />
          )}
        </Button>
      </div>
      {error && (
        <div
          className='bg-red-900/50 border border-red-700 text-red-100 px-4 py-3 m-4 rounded-md'
          role='alert'
        >
          <span className='block'>{error}</span>
        </div>
      )}
      <div className='flex flex-1 overflow-hidden'>
        <div
          id='graph-container'
          className='flex-1 overflow-auto'
          style={{ height: 'calc(100vh - 200px)' }}
          ref={graphContainerRef}
        >
          {nodes.length === 0 && (
            <p className='text-gray-500 italic text-center p-4'>
              Run a query to visualize the graph
            </p>
          )}
        </div>

        <div className='w-72 bg-gray-800 border-l border-gray-700 p-4 overflow-y-auto'>
          <Collapsible
            open={isNodesOpen}
            onOpenChange={setIsNodesOpen}
            className='mb-4'
          >
            <CollapsibleTrigger asChild>
              <Button variant='ghost' className='w-full justify-between'>
                <span>Node Types ({nodeTypes.length})</span>
                {isNodesOpen ? (
                  <ChevronUp className='h-4 w-4' />
                ) : (
                  <ChevronDown className='h-4 w-4' />
                )}
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className='bg-gray-700 rounded-md p-3 border border-gray-600'>
                {nodeTypes.length > 0 ? (
                  <div className='grid grid-cols-1 gap-2'>
                    {nodeTypes.map((type, index) => (
                      <div key={index} className='flex items-center'>
                        <div
                          className='w-3 h-3 mr-2 rounded-full'
                          style={{
                            backgroundColor: getColorForLabel(type) as string,
                          }}
                        />
                        <span className='text-gray-200 text-sm'>{type}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className='text-gray-400 text-sm italic'>
                    No nodes to display
                  </p>
                )}
              </div>
            </CollapsibleContent>
          </Collapsible>

          <Collapsible
            open={isRelationshipsOpen}
            onOpenChange={setIsRelationshipsOpen}
            className='mb-4'
          >
            <CollapsibleTrigger asChild>
              <Button variant='ghost' className='w-full justify-between'>
                <span>Relationship Types ({relationshipTypes.length})</span>
                {isRelationshipsOpen ? (
                  <ChevronUp className='h-4 w-4' />
                ) : (
                  <ChevronDown className='h-4 w-4' />
                )}
              </Button>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className='bg-gray-700 rounded-md p-3 border border-gray-600'>
                {relationshipTypes.length > 0 ? (
                  <div className='grid grid-cols-1 gap-2'>
                    {relationshipTypes.map((type, index) => (
                      <div key={index} className='flex items-center'>
                        <div className='w-8 h-[2px] mr-2 bg-blue-400'></div>
                        <span className='text-gray-200 text-sm'>{type}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className='text-gray-400 text-sm italic'>
                    No relationships to display
                  </p>
                )}
              </div>
            </CollapsibleContent>
          </Collapsible>
        </div>
      </div>
    </div>
  )
}
