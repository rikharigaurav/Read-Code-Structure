"use client";

import { useState, useEffect } from "react";
import { ResizablePanelGroup, ResizablePanel } from "@/components/ui/resizable";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { FileTree } from "@/components/file-tree";
import { IssuesList } from "@/components/issues-list";
import { CodeViewer } from "@/components/code-viewer";
import { ChatBot } from "@/components/chat-bot";
import { Neo4jGraph } from '@/components/neo4j-graph-visualizer'
import { useSearchParams } from "next/navigation";

export default function ExplorerPage() {
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [localFilePath, setLocalFilePath] = useState<string>('')
  const [repo, setRepo] = useState<string>('')
  const searchParams = useSearchParams()

  useEffect(() => {
    const pathFromUrl = searchParams.get('localFilePath')
    const repoFromUrl = searchParams.get('repo')
    if (pathFromUrl) {
      setLocalFilePath(pathFromUrl)
    }
    if (repoFromUrl) {
      setRepo(repoFromUrl)
    }
  }, [searchParams])
  // console.log('localFilePath:', localFilePath)
  // console.log('repo:', repo)
  return (
    <div className='h-screen bg-background'>
      <ResizablePanelGroup direction='horizontal'>
        <ResizablePanel defaultSize={25} minSize={20} maxSize={40}>
          <div className='h-screen flex flex-col'>
            <div className='p-4 border-b'>
              <h2 className='text-lg font-semibold'>Repository Structure</h2>
            </div>
            <ScrollArea className='flex-1'>
              <div className='p-4'>
                <FileTree onSelect={setSelectedFile} rootPath={localFilePath} />
              </div>
            </ScrollArea>
          </div>
        </ResizablePanel>

        <ResizablePanel defaultSize={75}>
          <Tabs defaultValue='issues' className='h-screen flex flex-col'>
            <div className='px-4 m-3 pt-2 border-b '>
              <TabsList className="flex justify-between">
                <TabsTrigger value='issues'>Issues</TabsTrigger>
                <TabsTrigger value='code'>Code Viewer</TabsTrigger>
                <TabsTrigger value='chat'>Chat Bot</TabsTrigger>
                <TabsTrigger value='neo4j'>CodeBase Graph</TabsTrigger>
              </TabsList>
            </div>

            <div className='flex-1 p-4'>
              <TabsContent value='issues' className='h-full'>
                <IssuesList repoURL={repo} />
              </TabsContent>

              <TabsContent value='code' className='h-full'>
                <CodeViewer file={selectedFile} />
              </TabsContent>

              <TabsContent value='chat' className='h-full'>
                <ChatBot />
              </TabsContent>

              <TabsContent value='neo4j' className='h-full'>
                <Neo4jGraph
                  uri={process.env.NEXT_PUBLIC_NEO4J_URI!}
                  user={process.env.NEXT_PUBLIC_NEO4J_USER!}
                  password={process.env.NEXT_PUBLIC_NEO4J_PASSWORD!}
                />
              </TabsContent>
            </div>
          </Tabs>
        </ResizablePanel>
      </ResizablePanelGroup>
    </div>
  )
}