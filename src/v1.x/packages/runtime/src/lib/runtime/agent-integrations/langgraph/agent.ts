import { Observable, Subscriber, map } from "rxjs";
import { LangGraphEventTypes } from "../../../../agents/langgraph/events";
import { RawEvent } from "@ag-ui/core";
import {
  LangGraphAgent as AGUILangGraphAgent,
  LangGraphHttpAgent,
  type LangGraphAgentConfig,
  ProcessedEvents,
  SchemaKeys,
  type State,
  StateEnrichment,
} from "@ag-ui/langgraph";
import { Message as LangGraphMessage } from "@langchain/langgraph-sdk/dist/types.messages";
import { StreamMode, ThreadState } from "@langchain/langgraph-sdk";

interface CopilotKitStateEnrichment {
  copilotkit: {
    actions: StateEnrichment["ag-ui"]["tools"];
    context: StateEnrichment["ag-ui"]["context"];
  };
}

import { RunAgentInput, EventType, CustomEvent, randomUUID, type ToolCall } from "@ag-ui/client";

type LangGraphContentPart =
  | { type: "text"; text: string }
  | { type: "image_url"; image_url: string | { url?: string } }
  | { type: string; [key: string]: unknown };

type AguiContentPart =
  | { type: "text"; text: string }
  | { type: "binary"; mimeType: string; data?: string; url?: string; id?: string };

type CopilotMessage = RunAgentInput["messages"][number];
type AguiToolCall = ToolCall;
type AguiMessage = CopilotMessage;

type LangGraphToolCall = {
  id: string;
  name: string;
  args: unknown;
};

type LangGraphMessageLike = {
  id?: string;
  role?: string;
  type?: string;
  content?: unknown;
  tool_calls?: LangGraphToolCall[];
  tool_call_id?: string;
  toolCallId?: string;
  name?: string;
};

type LangGraphRunSummary = {
  run_id: string;
  status?: string;
  created_at?: string;
  updated_at?: string;
  metadata?: {
    source?: string;
  };
};

type LangGraphAssistant = {
  assistant_id: string;
};

type LangGraphClient = {
  threads: {
    getState: (threadId: string) => Promise<ThreadState<State>>;
  };
  runs: {
    list: (
      threadId: string,
      options: { limit: number; offset: number },
    ) => Promise<LangGraphRunSummary[]>;
    joinStream: (
      threadId: string,
      runId: string,
      options: { streamMode?: StreamMode | StreamMode[]; lastEventId?: string },
    ) => unknown;
  };
  assistants: {
    getGraph: (assistantId: string) => Promise<unknown>;
  };
};

type LangGraphAgentInternals = {
  client: LangGraphClient;
  assistant?: LangGraphAssistant;
  activeRun?: {
    id: string;
    threadId?: string;
    hasFunctionStreaming?: boolean;
    connectRunStarted?: boolean;
    threadStream?: boolean;
    schemaKeys?: SchemaKeys;
    graphInfo?: unknown;
  };
  cancelRequested: boolean;
  cancelSent: boolean;
  subscriber: Subscriber<ProcessedEvents>;
};

type ConnectActiveRun = NonNullable<LangGraphAgentInternals["activeRun"]>;
type LangGraphPreparedStream = Awaited<
  | ReturnType<AGUILangGraphAgent["prepareStream"]>
  | ReturnType<AGUILangGraphAgent["prepareRegenerateStream"]>
>;
type LangGraphStreamResponse = Extract<LangGraphPreparedStream, { streamResponse: unknown }>["streamResponse"];

type LangGraphConnectForwardedProps = {
  streamMode?: StreamMode | StreamMode[];
  lastEventId?: string | number;
  connectPollIntervalMs?: number;
  connectRunListLimit?: number;
  connectRunWindowSize?: number;
  config?: {
    configurable?: {
      thread_id?: string;
    };
  };
};

type LangGraphRunForwardedProps = {
  streamMode?: StreamMode | StreamMode[];
  command?: unknown;
  regenerate?: boolean;
};

type RunAgentInputWithConfig = RunAgentInput & {
  config?: {
    configurable?: {
      thread_id?: string;
    };
  };
};

const DEFAULT_CONNECT_STREAM_MODES: StreamMode[] = ["events", "values", "updates"];
const DEFAULT_RUN_STREAM_MODES: StreamMode[] = ["events", "values", "updates"];
const DEFAULT_CONNECT_POLL_INTERVAL_MS = 1000;
const DEFAULT_CONNECT_RUN_LIST_LIMIT = 1000;
const DEFAULT_CONNECT_RUN_WINDOW_SIZE = 50;
const DEFAULT_BINARY_MIME_TYPE = "image/png";

const isImageUrlPart = (
  part: LangGraphContentPart,
): part is { type: "image_url"; image_url: string | { url?: string } } =>
  part.type === "image_url" && "image_url" in part;

const normalizeContentParts = (parts: LangGraphContentPart[]): AguiContentPart[] => {
  const normalized: AguiContentPart[] = [];

  for (const part of parts) {
    if (part.type === "text" && typeof part.text === "string") {
      normalized.push({ type: "text", text: part.text });
      continue;
    }

    if (isImageUrlPart(part)) {
      const imageUrl = typeof part.image_url === "string" ? part.image_url : part.image_url?.url ?? "";
      if (!imageUrl) {
        continue;
      }

      if (imageUrl.startsWith("data:")) {
        const [prefix, data] = imageUrl.split(",", 2);
        const mimeType = prefix.includes(":")
          ? prefix.split(":")[1].split(";")[0]
          : DEFAULT_BINARY_MIME_TYPE;
        normalized.push({ type: "binary", mimeType, data: data ?? "" });
        continue;
      }

      normalized.push({ type: "binary", mimeType: DEFAULT_BINARY_MIME_TYPE, url: imageUrl });
    }
  }

  return normalized;
};

const normalizeAguiContentParts = (parts: AguiContentPart[]): LangGraphContentPart[] => {
  const normalized: LangGraphContentPart[] = [];

  for (const part of parts) {
    if (part.type === "text" && typeof part.text === "string") {
      normalized.push({ type: "text", text: part.text });
      continue;
    }

    if (part.type === "binary") {
      let imageUrl: string | undefined;
      if (typeof part.url === "string" && part.url.length > 0) {
        imageUrl = part.url;
      } else if (typeof part.data === "string" && part.data.length > 0) {
        const mimeType =
          typeof part.mimeType === "string" && part.mimeType.length > 0
            ? part.mimeType
            : DEFAULT_BINARY_MIME_TYPE;
        imageUrl = `data:${mimeType};base64,${part.data}`;
      } else if (typeof part.id === "string" && part.id.length > 0) {
        imageUrl = part.id;
      }

      if (!imageUrl) {
        continue;
      }

      normalized.push({ type: "image_url", image_url: { url: imageUrl } });
    }
  }

  return normalized;
};

const extractTextContent = (content: unknown): string | null => {
  if (!content) {
    return null;
  }

  if (typeof content === "string") {
    return content;
  }

  if (Array.isArray(content)) {
    const textPart = content.find((part) => {
      if (!part || typeof part !== "object") {
        return false;
      }
      const candidate = part as { type?: unknown; text?: unknown };
      return candidate.type === "text" && typeof candidate.text === "string";
    }) as { text?: string } | undefined;

    return typeof textPart?.text === "string" ? textPart.text : null;
  }

  return null;
};

const stringifyContent = (content: unknown): string => {
  if (typeof content === "string") {
    return content;
  }
  return JSON.stringify(content);
};

const toLangGraphUserMessage = (message: CopilotMessage): LangGraphMessage | null => {
  if (message.role !== "user") {
    return null;
  }

  const contentValue = message.content;
  const content =
    typeof contentValue === "string"
      ? contentValue
      : Array.isArray(contentValue)
        ? normalizeAguiContentParts(contentValue as AguiContentPart[])
        : stringifyContent(contentValue);

  return {
    id: message.id,
    type: "human",
    content,
  } as LangGraphMessage;
};

const isRegenerateCommand = (command: unknown): boolean => {
  if (command === "regenerate") {
    return true;
  }

  if (!command || typeof command !== "object") {
    return false;
  }

  if (!("regenerate" in command)) {
    return false;
  }

  const { regenerate } = command as { regenerate?: unknown };
  return regenerate === true;
};

const normalizeLangGraphMessages = (
  messages: Array<LangGraphMessage | LangGraphMessageLike>,
): RunAgentInput["messages"] =>
  messages.reduce<AguiMessage[]>((normalized, message, index) => {
    const messageTypeCandidate = (message as { type?: unknown }).type;
    const roleCandidate = (message as { role?: unknown }).role;
    const messageType =
      typeof messageTypeCandidate === "string"
        ? messageTypeCandidate
        : typeof roleCandidate === "string"
          ? roleCandidate
          : undefined;

    const messageContent = (message as { content?: unknown }).content;

    switch (messageType) {
      case "human":
      case "user": {
        const content = Array.isArray(messageContent)
          ? normalizeContentParts(messageContent as LangGraphContentPart[])
          : stringifyContent(extractTextContent(messageContent));
        if (typeof message.id === "string") {
          normalized.push({ id: message.id, role: "user", content });
        }
        return normalized;
      }
      case "ai":
      case "assistant": {
        const textContent = extractTextContent(messageContent);
        const content = textContent ? stringifyContent(textContent) : "";
        const toolCallPayloads = (message as { tool_calls?: LangGraphToolCall[] }).tool_calls;
        const toolCalls = Array.isArray(toolCallPayloads)
          ? toolCallPayloads.map(
              (toolCall): AguiToolCall => ({
                id: toolCall.id,
                type: "function",
                function: {
                  name: toolCall.name,
                  arguments: JSON.stringify(toolCall.args),
                },
              }),
            )
          : undefined;
        if (typeof message.id === "string") {
          normalized.push({
            id: message.id,
            role: "assistant",
            content,
            ...(toolCalls ? { toolCalls } : {}),
          });
        }
        return normalized;
      }
      case "system": {
        const content = stringifyContent(extractTextContent(messageContent));
        if (typeof message.id === "string") {
          normalized.push({ id: message.id, role: "system", content });
        }
        return normalized;
      }
      case "tool": {
        const content = stringifyContent(extractTextContent(messageContent));
        if (typeof message.id === "string") {
          const toolCallId =
            (message as { tool_call_id?: string; toolCallId?: string }).tool_call_id ??
            (message as { toolCallId?: string }).toolCallId;
          normalized.push({
            id: message.id,
            role: "tool",
            content,
            toolCallId,
          });
        }
        return normalized;
      }
      case "function": {
        const content = stringifyContent(extractTextContent(messageContent));
        const toolCallId =
          typeof message.id === "string"
            ? message.id
            : message.name
              ? message.name
              : `function:${index}`;
        normalized.push({
          id: toolCallId,
          role: "tool",
          content,
          toolCallId,
        });
        return normalized;
      }
      case "remove": {
        const removeId = typeof message.id === "string" ? message.id : null;
        if (!removeId) {
          return normalized;
        }
        return normalized.filter((existing) => existing.id !== removeId);
      }
      default:
        throw new Error("message type returned from LangGraph is not supported.");
    }
  }, []);

const sleep = (durationMs: number) => new Promise((resolve) => setTimeout(resolve, durationMs));

const getErrorStatus = (error: unknown): number | undefined => {
  if (!error || typeof error !== "object") {
    return undefined;
  }

  const statusCandidate = (error as { status?: unknown }).status;
  if (typeof statusCandidate === "number") {
    return statusCandidate;
  }

  const statusCodeCandidate = (error as { statusCode?: unknown }).statusCode;
  if (typeof statusCodeCandidate === "number") {
    return statusCodeCandidate;
  }

  const responseCandidate = (error as { response?: unknown }).response;
  if (!responseCandidate || typeof responseCandidate !== "object") {
    return undefined;
  }

  const responseStatus = (responseCandidate as { status?: unknown }).status;
  return typeof responseStatus === "number" ? responseStatus : undefined;
};

const isNotFoundError = (error: unknown) => getErrorStatus(error) === 404;

// Import and re-export from separate file to maintain API compatibility
import { CustomEventNames, TextMessageEvents, ToolCallEvents, PredictStateTool } from "./consts";
export { CustomEventNames };

export class LangGraphAgent extends AGUILangGraphAgent {
  constructor(config: LangGraphAgentConfig) {
    super(config);
  }

  // @ts-ignore
  public clone() {
    return new LangGraphAgent(this.config);
  }

  dispatchEvent(event: ProcessedEvents) {
    if (event.type === EventType.CUSTOM) {
      // const event = processedEvent as unknown as CustomEvent;
      const customEvent = event as unknown as CustomEvent;

      if (customEvent.name === CustomEventNames.CopilotKitManuallyEmitMessage) {
        this.subscriber.next({
          type: EventType.TEXT_MESSAGE_START,
          role: "assistant",
          messageId: customEvent.value.message_id,
          rawEvent: event,
        });
        this.subscriber.next({
          type: EventType.TEXT_MESSAGE_CONTENT,
          messageId: customEvent.value.message_id,
          delta: customEvent.value.message,
          rawEvent: event,
        });
        this.subscriber.next({
          type: EventType.TEXT_MESSAGE_END,
          messageId: customEvent.value.message_id,
          rawEvent: event,
        });
        return true;
      }

      if (customEvent.name === CustomEventNames.CopilotKitManuallyEmitToolCall) {
        this.subscriber.next({
          type: EventType.TOOL_CALL_START,
          toolCallId: customEvent.value.id,
          toolCallName: customEvent.value.name,
          parentMessageId: customEvent.value.id,
          rawEvent: event,
        });
        this.subscriber.next({
          type: EventType.TOOL_CALL_ARGS,
          toolCallId: customEvent.value.id,
          delta: customEvent.value.args,
          rawEvent: event,
        });
        this.subscriber.next({
          type: EventType.TOOL_CALL_END,
          toolCallId: customEvent.value.id,
          rawEvent: event,
        });
        return true;
      }

      if (customEvent.name === CustomEventNames.CopilotKitManuallyEmitIntermediateState) {
        this.activeRun.manuallyEmittedState = customEvent.value;
        this.dispatchEvent({
          type: EventType.STATE_SNAPSHOT,
          snapshot: this.getStateSnapshot({
            values: this.activeRun.manuallyEmittedState,
          } as ThreadState<State>),
          rawEvent: event,
        });
        return true;
      }

      if (customEvent.name === CustomEventNames.CopilotKitExit) {
        this.subscriber.next({
          type: EventType.CUSTOM,
          name: "Exit",
          value: true,
        });
        return true;
      }
    }

    // Intercept all text message and tool call events and check if should disable
    const rawEvent = (event as ToolCallEvents | TextMessageEvents).rawEvent;
    if (!rawEvent) {
      this.subscriber.next(event);
      return true;
    }

    const isMessageEvent =
      event.type === EventType.TEXT_MESSAGE_START ||
      event.type === EventType.TEXT_MESSAGE_CONTENT ||
      event.type === EventType.TEXT_MESSAGE_END;
    const isToolEvent =
      event.type === EventType.TOOL_CALL_START ||
      event.type === EventType.TOOL_CALL_ARGS ||
      event.type === EventType.TOOL_CALL_END;
    if ("copilotkit:emit-tool-calls" in (rawEvent.metadata || {})) {
      if (rawEvent.metadata["copilotkit:emit-tool-calls"] === false && isToolEvent) {
        return false;
      }
    }
    if ("copilotkit:emit-messages" in (rawEvent.metadata || {})) {
      if (rawEvent.metadata["copilotkit:emit-messages"] === false && isMessageEvent) {
        return false;
      }
    }

    this.subscriber.next(event);
    return true;
  }

  // @ts-ignore
  run(input: RunAgentInput) {
    return super.run(input).pipe(
      map((processedEvent) => {
        // Turn raw event into emit state snapshot from tool call event
        if (processedEvent.type === EventType.RAW) {
          // Get the LangGraph event from the AGUI event.
          const event = (processedEvent as RawEvent).event ?? (processedEvent as RawEvent).rawEvent;

          const eventType = event.event;
          const toolCallData = event.data?.chunk?.tool_call_chunks?.[0];
          const toolCallUsedToPredictState = event.metadata?.[
            "copilotkit:emit-intermediate-state"
          ]?.some(
            (predictStateTool: PredictStateTool) => predictStateTool.tool === toolCallData?.name,
          );

          if (eventType === LangGraphEventTypes.OnChatModelStream && toolCallUsedToPredictState) {
            return {
              type: EventType.CUSTOM,
              name: "PredictState",
              value: event.metadata["copilotkit:emit-intermediate-state"],
            };
          }
        }

        return processedEvent;
      }),
    );
  }

  async runAgentStream(
    input: Parameters<AGUILangGraphAgent["runAgentStream"]>[0],
    subscriber: Subscriber<ProcessedEvents>,
  ): Promise<void> {
    const forwardedProps = input.forwardedProps as LangGraphRunForwardedProps | undefined;
    const regenerateRequested =
      forwardedProps?.regenerate === true || isRegenerateCommand(forwardedProps?.command);

    if (!regenerateRequested) {
      return super.runAgentStream(input, subscriber);
    }

    const internals = this as unknown as LangGraphAgentInternals;
    internals.activeRun = {
      id: input.runId,
      threadId: input.threadId,
      hasFunctionStreaming: false,
    };
    internals.cancelRequested = false;
    internals.cancelSent = false;
    internals.subscriber = subscriber;

    if (!internals.assistant) {
      internals.assistant = (await this.getAssistant()) as LangGraphAssistant;
    }

    const threadId = input.threadId ?? randomUUID();
    const streamMode = forwardedProps?.streamMode ?? DEFAULT_RUN_STREAM_MODES;
    const lastUserMessage = [...input.messages].reverse().find((message) => message.role === "user");

    if (!lastUserMessage) {
      return subscriber.error("No user message found in messages to regenerate");
    }

    const messageCheckpoint = toLangGraphUserMessage(lastUserMessage);
    if (!messageCheckpoint) {
      return subscriber.error("No user message found in messages to regenerate");
    }

    const preparedStream = await this.prepareRegenerateStream(
      { ...(input as RunAgentInput), threadId, messageCheckpoint },
      streamMode,
    );

    if (!preparedStream) {
      return subscriber.error("No stream to regenerate");
    }

    await this.handleStreamEvents(
      preparedStream,
      threadId,
      subscriber,
      input,
      Array.isArray(streamMode) ? streamMode : [streamMode],
    );
  }

  protected connect(input: Parameters<AGUILangGraphAgent["run"]>[0]): Observable<ProcessedEvents> {
    return new Observable<ProcessedEvents>((subscriber) => {
      (async () => {
        const internals = this as unknown as LangGraphAgentInternals;
        internals.subscriber = subscriber;
        internals.cancelRequested = false;
        internals.cancelSent = false;

        const inputWithConfig = input as RunAgentInputWithConfig;
        let threadId =
          input.threadId ??
          inputWithConfig.config?.configurable?.thread_id ??
          (input.forwardedProps as LangGraphConnectForwardedProps | undefined)?.config?.configurable
            ?.thread_id;

        if (!threadId) {
          const error = new Error("Thread ID is required to connect");
          this.dispatchEvent({ type: EventType.RUN_ERROR, message: error.message });
          internals.cancelRequested = false;
          internals.cancelSent = false;
          internals.activeRun = undefined;
          subscriber.error(error);
          return;
        }

        const resolvedThreadId = threadId;

        const forwardedProps = input.forwardedProps as LangGraphConnectForwardedProps | undefined;
        const streamMode = forwardedProps?.streamMode ?? DEFAULT_CONNECT_STREAM_MODES;
        const streamModes = Array.isArray(streamMode) ? streamMode : [streamMode];
        const pollIntervalMs =
          forwardedProps?.connectPollIntervalMs ?? DEFAULT_CONNECT_POLL_INTERVAL_MS;
        const runListLimit = forwardedProps?.connectRunListLimit ?? DEFAULT_CONNECT_RUN_LIST_LIMIT;
        const runWindowSize =
          forwardedProps?.connectRunWindowSize ?? DEFAULT_CONNECT_RUN_WINDOW_SIZE;

        let lastEventId =
          forwardedProps?.lastEventId !== undefined ? String(forwardedProps.lastEventId) : undefined;

        const client = internals.client;
        const getThreadState = async () => {
          try {
            return await client.threads.getState(resolvedThreadId);
          } catch (error) {
            if (isNotFoundError(error)) {
              return {
                values: { messages: [] as LangGraphMessage[] },
              } as unknown as ThreadState<State>;
            }
            throw error;
          }
        };

        const assistant =
          internals.assistant ?? ((await this.getAssistant()) as LangGraphAssistant);
        internals.assistant = assistant;

        const initialState = await getThreadState();
        const initialRunId = input.runId ?? randomUUID();
        const initialActiveRun: ConnectActiveRun = {
          id: initialRunId,
          threadId: resolvedThreadId,
          hasFunctionStreaming: false,
          connectRunStarted: true,
          threadStream: true,
        };
        internals.activeRun = initialActiveRun;
        initialActiveRun.schemaKeys = await this.getSchemaKeys();
        initialActiveRun.graphInfo = await client.assistants.getGraph(assistant.assistant_id);

        this.dispatchEvent({ type: EventType.RUN_STARTED, threadId: resolvedThreadId, runId: initialRunId });
        this.dispatchEvent({
          type: EventType.STATE_SNAPSHOT,
          snapshot: this.getStateSnapshot(initialState),
        });

        const initialMessages = (initialState.values as { messages?: LangGraphMessage[] }).messages;
        if (initialMessages) {
          this.dispatchEvent({
            type: EventType.MESSAGES_SNAPSHOT,
            messages: normalizeLangGraphMessages(initialMessages),
          });
        }

        const initialInterrupts =
          (initialState as { tasks?: Array<{ interrupts?: Array<{ value?: unknown }> }> }).tasks?.[0]
            ?.interrupts ?? [];
        for (const interrupt of initialInterrupts) {
          this.dispatchEvent({
            type: EventType.CUSTOM,
            name: "on_interrupt",
            value: typeof interrupt.value === "string" ? interrupt.value : JSON.stringify(interrupt.value),
            rawEvent: interrupt,
          });
        }

        this.dispatchEvent({ type: EventType.RUN_FINISHED, threadId: resolvedThreadId, runId: initialRunId });

        let lastSeenRunId: string | undefined;
        let previousWindow: LangGraphRunSummary[] = [];

        const getRunTimestamp = (run: LangGraphRunSummary) =>
          Date.parse(run.updated_at ?? run.created_at ?? "") || 0;

        const pickLatestRun = (runs: LangGraphRunSummary[]) =>
          runs.length > 0
            ? runs.slice().sort((a, b) => getRunTimestamp(b) - getRunTimestamp(a))[0]
            : undefined;

        const emitSnapshotForRun = async (runId: string) => {
          const snapshotRun: ConnectActiveRun = {
            id: runId,
            threadId: resolvedThreadId,
            hasFunctionStreaming: false,
            connectRunStarted: true,
            threadStream: true,
          };
          internals.activeRun = snapshotRun;
          snapshotRun.schemaKeys = await this.getSchemaKeys();

          const snapshotState = await getThreadState();
          this.dispatchEvent({ type: EventType.RUN_STARTED, threadId: resolvedThreadId, runId });
          this.dispatchEvent({
            type: EventType.STATE_SNAPSHOT,
            snapshot: this.getStateSnapshot(snapshotState),
          });
          const snapshotMessages =
            (snapshotState.values as { messages?: LangGraphMessage[] }).messages ?? [];
          this.dispatchEvent({
            type: EventType.MESSAGES_SNAPSHOT,
            messages: normalizeLangGraphMessages(snapshotMessages),
          });
          this.dispatchEvent({ type: EventType.RUN_FINISHED, threadId: resolvedThreadId, runId });
          internals.activeRun = undefined;
        };

        while (!internals.cancelRequested) {
          let runs: LangGraphRunSummary[];

          try {
            runs = await client.runs.list(resolvedThreadId, { limit: runListLimit, offset: 0 });
          } catch (error) {
            if (isNotFoundError(error)) {
              await sleep(pollIntervalMs);
              continue;
            }
            throw error;
          }

          let window = runs.slice(Math.max(0, runs.length - runWindowSize));
          if (
            previousWindow.length > 0 &&
            window.length > 0 &&
            previousWindow[0].run_id === window[0].run_id &&
            previousWindow[previousWindow.length - 1]?.run_id === window[window.length - 1]?.run_id
          ) {
            window = previousWindow;
          } else {
            previousWindow = window;
          }

          const activeRuns = window.filter(
            (run) => run.status === "pending" || run.status === "running",
          );
          const activeRun = pickLatestRun(activeRuns);
          const latestRun = pickLatestRun(window);

          if (!activeRun) {
            if (latestRun && lastSeenRunId !== latestRun.run_id) {
              lastSeenRunId = latestRun.run_id;
              await emitSnapshotForRun(latestRun.run_id);
            }
            await sleep(pollIntervalMs);
            continue;
          }

          const activeRunSource = activeRun.metadata?.source;
          if (activeRunSource === "cron" || activeRunSource === "starter-cron") {
            if (lastSeenRunId !== activeRun.run_id) {
              lastSeenRunId = activeRun.run_id;
              await emitSnapshotForRun(activeRun.run_id);
            }
            await sleep(pollIntervalMs);
            continue;
          }

          if (lastSeenRunId === activeRun.run_id) {
            await sleep(pollIntervalMs);
            continue;
          }

          lastSeenRunId = activeRun.run_id;
          const streamingRun: ConnectActiveRun = {
            id: activeRun.run_id,
            threadId: resolvedThreadId,
            hasFunctionStreaming: false,
            connectRunStarted: true,
            threadStream: true,
          };
          internals.activeRun = streamingRun;
          streamingRun.schemaKeys = await this.getSchemaKeys();
          streamingRun.graphInfo = await client.assistants.getGraph(assistant.assistant_id);

          const streamState = await getThreadState();
          const stream = client.runs.joinStream(resolvedThreadId, activeRun.run_id, {
            streamMode,
            lastEventId,
          }) as LangGraphStreamResponse;
          lastEventId = undefined;

          const relay = new Subscriber<ProcessedEvents>({
            next: (event) => subscriber.next(event),
            error: (error) => subscriber.error(error),
            complete: () => {},
          });

          await this.handleStreamEvents(
            { streamResponse: stream, state: streamState },
            resolvedThreadId,
            relay,
            input,
            streamModes,
          );
        }

        internals.cancelRequested = false;
        internals.cancelSent = false;
        internals.activeRun = undefined;
        subscriber.complete();
      })().catch((error) => {
        const err = error instanceof Error ? error : new Error(String(error));
        const internals = this as unknown as LangGraphAgentInternals;
        this.dispatchEvent({ type: EventType.RUN_ERROR, message: err.message });
        internals.cancelRequested = false;
        internals.cancelSent = false;
        internals.activeRun = undefined;
        subscriber.error(err);
      });

      return () => {};
    });
  }

  langGraphDefaultMergeState(
    state: State,
    messages: LangGraphMessage[],
    input: RunAgentInput,
  ): State<StateEnrichment & CopilotKitStateEnrichment> {
    const aguiMergedState = super.langGraphDefaultMergeState(state, messages, input);
    const { tools: returnedTools, "ag-ui": agui } = aguiMergedState;
    // tolerate undefined and de-duplicate by stable key (id | name | key)
    const rawCombinedTools = [
      ...((returnedTools as any[]) ?? []),
      ...((agui?.tools as any[]) ?? []),
    ];
    const combinedTools = Array.from(
      new Map(
        rawCombinedTools.map((t: any) => [t?.id ?? t?.name ?? t?.key ?? JSON.stringify(t), t]),
      ).values(),
    );

    return {
      ...aguiMergedState,
      copilotkit: {
        actions: combinedTools,
        context: agui?.context ?? [],
      },
    };
  }

  async getSchemaKeys(): Promise<SchemaKeys> {
    const CONSTANT_KEYS = ["copilotkit"];

    try {
      const schemaResponse = await this.client.assistants.getSchemas(this.assistant.assistant_id);
      const contextSchema = (schemaResponse as { context_schema?: { properties?: object } })
        .context_schema;
      const contextKeys = contextSchema?.properties ? Object.keys(contextSchema.properties) : [];
      const configSchema = (schemaResponse as { config_schema?: { properties?: object } })
        .config_schema;
      const configKeys = configSchema?.properties ? Object.keys(configSchema.properties) : [];
      const inputSchema = (schemaResponse as { input_schema?: { properties?: object } }).input_schema;
      const outputSchema = (schemaResponse as { output_schema?: { properties?: object } }).output_schema;

      if (!inputSchema?.properties || !outputSchema?.properties) {
        return {
          config: configKeys,
          input: null,
          output: null,
          context: contextKeys.length ? [...contextKeys, ...CONSTANT_KEYS] : [],
        };
      }

      const inputKeys = Object.keys(inputSchema.properties);
      const outputKeys = Object.keys(outputSchema.properties);

      return {
        config: configKeys,
        input: inputKeys.length ? [...inputKeys, ...CONSTANT_KEYS] : null,
        output: outputKeys.length ? [...outputKeys, ...CONSTANT_KEYS] : null,
        context: contextKeys.length ? [...contextKeys, ...CONSTANT_KEYS] : [],
      };
    } catch {
      return {
        config: [],
        input: null,
        output: null,
        context: [],
      };
    }
  }
}

export { LangGraphHttpAgent };
