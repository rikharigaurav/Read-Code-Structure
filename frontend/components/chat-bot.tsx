"use client";

import { useState } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Send } from "lucide-react";
import { useChat } from "@/hooks/use-chat";
import type { ChatResponse } from "@/lib/types/api";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  structured_data?: ChatResponse['structured_data'];
}

export function ChatBot() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      role: "assistant",
      content: "Hello! I'm here to help you with your code questions. What would you like to know?",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const { sendMessage, isLoading, error } = useChat();

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");

    try {
      const result = await sendMessage(input);
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: result.content,
        timestamp: new Date(),
        structured_data: result.structured_data,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "Sorry, I encountered an error processing your request.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    }
  };

  return (
    <div className="h-full flex flex-col">
      <Card className="flex-1 mb-4">
        <ScrollArea className="h-[600px]">
          <div className="p-4 space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${
                  message.role === "user" ? "justify-end" : "justify-start"
                }`}
              >
                <div
                  className={`max-w-[80%] p-3 rounded-lg ${
                    message.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-accent"
                  }`}
                >
                  <p>{message.content}</p>
                  {message.structured_data && (
                    <div className="mt-2 space-y-2">
                      {message.structured_data.knowledge && (
                        <div className="text-sm bg-background/10 p-2 rounded">
                          <strong>Knowledge:</strong> {message.structured_data.knowledge}
                        </div>
                      )}
                      {message.structured_data.insights && (
                        <div className="text-sm bg-background/10 p-2 rounded">
                          <strong>Insights:</strong>
                          <ul className="list-disc list-inside mt-1">
                            {message.structured_data.insights.map((insight, i) => (
                              <li key={i}>{insight}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {message.structured_data.code && (
                        <div className="text-sm bg-background/10 p-2 rounded font-mono">
                          <strong>Code:</strong>
                          <pre className="mt-1 overflow-x-auto">
                            {message.structured_data.code}
                          </pre>
                        </div>
                      )}
                    </div>
                  )}
                  <p className="text-xs text-muted-foreground mt-1">
                    {message.timestamp.toLocaleTimeString()}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </Card>

      <div className="flex gap-2">
        <Input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          onKeyPress={(e) => e.key === "Enter" && !isLoading && handleSend()}
          disabled={isLoading}
        />
        <Button onClick={handleSend} disabled={isLoading}>
          <Send className="w-4 h-4" />
        </Button>
      </div>
    </div>
  );
}