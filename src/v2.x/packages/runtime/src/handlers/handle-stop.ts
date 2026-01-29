import { CopilotRuntime } from "../runtime";
import { EventType } from "@ag-ui/client";
import { CONNECT_ABORTERS_BY_THREAD } from "./connect-aborters";

interface StopAgentParameters {
  request: Request;
  runtime: CopilotRuntime;
  agentId: string;
  threadId: string;
}

export async function handleStopAgent({
  runtime,
  request,
  agentId,
  threadId,
}: StopAgentParameters) {
  try {
    const agents = await runtime.agents;

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

    const connectKey = `${agentId}:${threadId}`;
    const aborters = CONNECT_ABORTERS_BY_THREAD.get(connectKey);
    if (aborters && aborters.size > 0) {
      for (const aborter of Array.from(aborters)) {
        try {
          aborter();
        } catch {
          // ignore abort errors
        }
      }
    }

    const stopped = await runtime.runner.stop({ threadId });

    if (!stopped) {
      return new Response(
        JSON.stringify({
          stopped: false,
          message: `No active run for thread '${threadId}'.`,
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }
      );
    }

    return new Response(
      JSON.stringify({
        stopped: true,
        interrupt: {
          type: EventType.RUN_ERROR,
          message: "Run stopped by user",
          code: "STOPPED",
        },
      }),
      {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }
    );
  } catch (error) {
    console.error("Error stopping agent run:", error);

    return new Response(
      JSON.stringify({
        error: "Failed to stop agent",
        message: error instanceof Error ? error.message : "Unknown error",
      }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      }
    );
  }
}
