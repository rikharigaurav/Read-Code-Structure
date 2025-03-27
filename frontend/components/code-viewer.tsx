"use client";

import { ScrollArea } from "@/components/ui/scroll-area";
import { Card } from "@/components/ui/card";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";

interface CodeViewerProps {
  file: string | null;
}

export function CodeViewer({ file }: CodeViewerProps) {
  // Placeholder code - replace with actual file contents from API
  const code = file
    ? `// File: ${file}\n\nconst example = "This is a sample code file";\nconsole.log(example);`
    : null;

  return (
    <Card className="h-full">
      <ScrollArea className="h-[600px]">
        <div className="p-4">
          {code ? (
            <SyntaxHighlighter
              language="typescript"
              style={vscDarkPlus}
              customStyle={{
                margin: 0,
                borderRadius: "0.5rem",
              }}
            >
              {code}
            </SyntaxHighlighter>
          ) : (
            <div className="text-center text-muted-foreground">
              Select a file to view its contents
            </div>
          )}
        </div>
      </ScrollArea>
    </Card>
  );
}