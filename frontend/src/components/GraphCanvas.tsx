"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import ForceGraph2D, { type ForceGraphMethods, type LinkObject, type NodeObject } from "react-force-graph-2d";
import type { EntityType, GraphData, GraphLink, GraphNode } from "@/lib/api";

type Node = NodeObject<GraphNode>;
type Link = LinkObject<GraphNode, GraphLink>;

const TYPES: EntityType[] = ["Method", "Model", "Dataset", "Metric", "Task", "Concept", "Tool"];

function readPalette() {
  const css = getComputedStyle(document.documentElement);
  const get = (name: string) => css.getPropertyValue(name).trim();
  return {
    types: Object.fromEntries(TYPES.map((t) => [t, get(`--type-${t.toLowerCase()}`)])) as Record<EntityType, string>,
    ink: get("--ink"),
    muted: get("--ink-faint"),
    line: get("--line-strong"),
    highlight: get("--highlight"),
    surface: get("--surface"),
    font: getComputedStyle(document.body).fontFamily,
  };
}

function usePalette() {
  const [palette, setPalette] = useState<ReturnType<typeof readPalette> | null>(null);
  useEffect(() => {
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const update = () => setPalette(readPalette());
    update();
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  return palette;
}

interface Props {
  data: GraphData;
  width: number;
  height: number;
  selected: string | null;
  onSelect: (node: GraphNode | null) => void;
}

export default function GraphCanvas({ data, width, height, selected, onSelect }: Props) {
  const palette = usePalette();
  const graphRef = useRef<ForceGraphMethods<Node, Link> | undefined>(undefined);

  // The library mutates nodes and links in place, so hand it its own copy.
  const graphData = useMemo(
    () => ({ nodes: data.nodes.map((n) => ({ ...n })), links: data.links.map((l) => ({ ...l })) }),
    [data],
  );

  const neighbours = useMemo(() => {
    const map = new Map<string, Set<string>>();
    for (const link of data.links) {
      if (!map.has(link.source)) map.set(link.source, new Set());
      if (!map.has(link.target)) map.set(link.target, new Set());
      map.get(link.source)!.add(link.target);
      map.get(link.target)!.add(link.source);
    }
    return map;
  }, [data]);

  useEffect(() => {
    const timer = setTimeout(() => graphRef.current?.zoomToFit(400, 40), 600);
    return () => clearTimeout(timer);
  }, [graphData, width, height]);

  if (!palette) return null;

  const isFocused = (id: string) => !selected || id === selected || neighbours.get(selected)?.has(id);
  const endpointId = (end: Link["source"]) => (typeof end === "object" ? (end as Node).id : end) as string;

  return (
    <ForceGraph2D<GraphNode, GraphLink>
      ref={graphRef}
      graphData={graphData}
      width={width}
      height={height}
      backgroundColor="rgba(0,0,0,0)"
      cooldownTicks={120}
      linkDirectionalArrowLength={3}
      linkDirectionalArrowRelPos={0.92}
      linkColor={(link) => {
        const focused = isFocused(endpointId(link.source)) && isFocused(endpointId(link.target));
        if (!focused) return `${palette.line}33`;
        return link.bridge ? palette.ink : palette.line;
      }}
      linkWidth={(link) => (link.bridge ? 1.8 : 0.6 + Math.min(link.weight ?? 1, 4) * 0.3)}
      linkLabel={(link) => `${link.type.replaceAll("_", " ").toLowerCase()}: ${link.description}`}
      nodeLabel={(node) => `${node.name} (${node.type})`}
      onNodeClick={(node) => onSelect(node.id === selected ? null : node)}
      onBackgroundClick={() => onSelect(null)}
      nodePointerAreaPaint={(node, color, ctx) => {
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(node.x!, node.y!, radius(node) + 3, 0, 2 * Math.PI);
        ctx.fill();
      }}
      nodeCanvasObject={(node, ctx, scale) => {
        const r = radius(node);
        const dim = !isFocused(node.id);
        ctx.globalAlpha = dim ? 0.18 : 1;
        if (node.seed) {
          ctx.beginPath();
          ctx.arc(node.x!, node.y!, r + 3.5, 0, 2 * Math.PI);
          ctx.fillStyle = palette.highlight;
          ctx.fill();
        }
        ctx.beginPath();
        ctx.arc(node.x!, node.y!, r, 0, 2 * Math.PI);
        ctx.fillStyle = palette.types[node.type] || palette.muted;
        ctx.fill();
        if (node.id === selected) {
          ctx.lineWidth = 1.5;
          ctx.strokeStyle = palette.ink;
          ctx.stroke();
        }
        const showLabel = node.seed || node.id === selected || scale > 1.6 || (node.degree ?? 0) >= 6;
        if (showLabel) {
          const size = Math.max(11 / scale, 2.5);
          ctx.font = `${node.seed ? 600 : 500} ${size}px ${palette.font}`;
          ctx.textAlign = "center";
          ctx.textBaseline = "top";
          ctx.lineWidth = 3 / scale;
          ctx.strokeStyle = palette.surface;
          ctx.strokeText(node.name, node.x!, node.y! + r + 2);
          ctx.fillStyle = palette.ink;
          ctx.fillText(node.name, node.x!, node.y! + r + 2);
        }
        ctx.globalAlpha = 1;
      }}
    />
  );
}

function radius(node: Node): number {
  const weight = node.degree ?? node.mentions ?? (node.seed ? 6 : 2);
  return 3 + Math.sqrt(weight) * 1.4;
}
