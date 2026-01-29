import { AbstractAgent, RunAgentInput, RunAgentInputSchema } from "@ag-ui/client";
import { EventEncoder } from "@ag-ui/encoder";
import { Observable, type Subscription } from "rxjs";
import { CopilotRuntime } from "../runtime";
import { extractForwardableHeaders } from "./header-utils";
import { CONNECT_ABORTERS_BY_THREAD } from "./connect-aborters";

interface ConnectAgentParameters {
  request: Request;
  runtime: CopilotRuntime;
  agentId: string;
}

export async function handleConnectAgent({
  runtime,
  request,
  agentId,
}: ConnectAgentParameters) {
  try {
    const agents = await runtime.agents;

    // Check if the requested agent exists
    if (!agents[agentId]) {
      return new Response(
        JSON.stringify({
          error: "Agent not found",
          message: `Agent '${agentId}' does not exist`,
        }),
        {
          status: 404,
          headers: { "Content-Type": "application/json" },
        }
      );
    }

    const stream = new TransformStream();
    const writer = stream.writable.getWriter();
    const encoder = new EventEncoder();
    let streamClosed = false;
    let unregisterAborter = () => {};

    // Process the request in the background
    (async () => {
      let input: RunAgentInput;
      try {
        const requestBody = await request.json();
        input = RunAgentInputSchema.parse(requestBody);
      } catch {
        return new Response(
          JSON.stringify({
            error: "Invalid request body",
          }),
          { status: 400 }
        );
      }

      const forwardableHeaders = extractForwardableHeaders(request);

      const registeredAgent = agents[agentId] as AbstractAgent;
      const agent =
        typeof registeredAgent.clone === "function"
          ? (registeredAgent.clone() as AbstractAgent)
          : registeredAgent;

      if (agent && "headers" in agent) {
        agent.headers = {
          ...(agent.headers as Record<string, string>),
          ...forwardableHeaders,
        };
      }

      if (agent && typeof agent.setMessages === "function") {
        agent.setMessages(input.messages);
      }
      if (agent && typeof agent.setState === "function") {
        agent.setState(input.state);
      }
      if (agent) {
        agent.threadId = input.threadId;
      }

      const connectKey = `${agentId}:${input.threadId}`;
      const connectAgent = agent as AbstractAgent & {
        connect?: (input: RunAgentInput) => Observable<unknown>;
        abortRun?: () => void;
      };
      const hasConnect = typeof connectAgent.connect === "function";
      const events$ = hasConnect
        ? (connectAgent.connect as (input: RunAgentInput) => Observable<unknown>)(input)
        : runtime.runner.connect({
            threadId: input.threadId,
            headers: forwardableHeaders,
          });

      let abortRequested = false;
      let subscription: Subscription | undefined;

      const unregisterAborterInner = () => {
        const aborters = CONNECT_ABORTERS_BY_THREAD.get(connectKey);
        if (!aborters) {
          return;
        }
        aborters.delete(abortHandler);
        if (aborters.size === 0) {
          CONNECT_ABORTERS_BY_THREAD.delete(connectKey);
        }
      };

      unregisterAborter = unregisterAborterInner;

      const abortHandler = () => {
        if (abortRequested) {
          return;
        }
        abortRequested = true;
        unregisterAborterInner();

        if (connectAgent && typeof connectAgent.abortRun === "function") {
          try {
            connectAgent.abortRun();
          } catch {
            // ignore abort errors
          }
        }

        if (subscription) {
          try {
            subscription.unsubscribe();
          } catch {
            // ignore unsubscribe errors
          }
        }

        if (!streamClosed) {
          streamClosed = true;
          try {
            writer.close();
          } catch {
            // Stream already closed
          }
        }
      };

      const registerAborter = () => {
        const existing = CONNECT_ABORTERS_BY_THREAD.get(connectKey);
        const aborters = existing ?? new Set();
        if (!existing) {
          CONNECT_ABORTERS_BY_THREAD.set(connectKey, aborters);
        }
        aborters.add(abortHandler);
      };

      registerAborter();

      subscription = events$.subscribe({
          next: async (event) => {
            if (!request.signal.aborted && !streamClosed) {
              try {
                await writer.write(encoder.encode(event));
              } catch (error) {
                if (error instanceof Error && error.name === "AbortError") {
                  abortHandler();
                }
              }
            }
          },
          error: async (error) => {
            console.error("Error running agent:", error);
            if (!streamClosed) {
              try {
                await writer.close();
                streamClosed = true;
              } catch {
                // Stream already closed
              }
            }
            unregisterAborter();
            if (request.signal) {
              request.signal.removeEventListener("abort", abortHandler);
            }
          },
          complete: async () => {
            if (!streamClosed) {
              try {
                await writer.close();
                streamClosed = true;
              } catch {
                // Stream already closed
              }
            }
            unregisterAborter();
            if (request.signal) {
              request.signal.removeEventListener("abort", abortHandler);
            }
          },
        });

      if (request.signal) {
        request.signal.addEventListener("abort", abortHandler);
        if (request.signal.aborted) {
          abortHandler();
        }
      }
    })().catch((error) => {
      console.error("Error running agent:", error);
      console.error(
        "Error stack:",
        error instanceof Error ? error.stack : "No stack trace"
      );
      console.error("Error details:", {
        name: error instanceof Error ? error.name : "Unknown",
        message: error instanceof Error ? error.message : String(error),
        cause: error instanceof Error ? error.cause : undefined,
      });
      unregisterAborter();
      if (!streamClosed) {
        try {
          writer.close();
          streamClosed = true;
        } catch {
          // Stream already closed
        }
      }
    });

    // Return the SSE response
    return new Response(stream.readable, {
      status: 200,
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      },
    });
  } catch (error) {
    console.error("Error running agent:", error);
    console.error(
      "Error stack:",
      error instanceof Error ? error.stack : "No stack trace"
    );
    console.error("Error details:", {
      name: error instanceof Error ? error.name : "Unknown",
      message: error instanceof Error ? error.message : String(error),
      cause: error instanceof Error ? error.cause : undefined,
    });

    return new Response(
      JSON.stringify({
        error: "Failed to run agent",
        message: error instanceof Error ? error.message : "Unknown error",
      }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
}
