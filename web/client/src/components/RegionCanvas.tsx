import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { DetectedFaceRecord } from "../api";
import {
  POSE_LIMB_GROUPS,
  POSE_CONNECTIONS,
  POSE_JOINT_NAMES,
  POSE_TORSO_JOINTS,
  poseGroupCenter,
  rotatePoseGroup,
  standingPose,
  translatePoseGroup,
  type PoseLimbName,
  type PoseJointName,
  type SubjectPoseState,
} from "../pose.ts";
import { Icon } from "./Icon";

export type StudioMode = "generation" | "edit" | "face";
export type RegionLayer = "generation" | "reference" | "targets";
export type DrawMode = "region" | "subject" | null;

export interface RegionBox {
  id: string;
  name: string;
  layer: RegionLayer;
  x: number;
  y: number;
  width: number;
  height: number;
  prompt: string;
  faceIdentityPrompt: string;
  spatialRole: "auto" | "subject" | "background";
  regionType: "region" | "subject";
  pose: SubjectPoseState | null;
  enabled: boolean;
  priority?: number;
}

type ResizeEdge = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";

interface DragState {
  kind: "draw" | "move" | "resize" | "lasso" | "joint" | "head-move" | "head-resize-x" | "head-resize-y" | "pose-group-move" | "torso-rotate";
  regionId?: string;
  jointName?: PoseJointName;
  poseGroup?: PoseLimbName | "torso";
  edge?: ResizeEdge;
  startX: number;
  startY: number;
  initial?: RegionBox;
}

interface Props {
  mode: StudioMode;
  activeLayer: RegionLayer;
  sourceUrl: string | null;
  sourceName: string;
  resultUrl: string | null;
  resultName: string;
  regions: RegionBox[];
  selectedId: string | null;
  drawMode: DrawMode;
  comparePosition: number;
  canvasWidth: number;
  canvasHeight: number;
  faces: DetectedFaceRecord[];
  selectedFaceIndices: number[];
  manualFacePaths: number[][][];
  lassoMode: boolean;
  onComparePosition: (value: number) => void;
  onSelect: (id: string | null) => void;
  onRegions: (regions: RegionBox[]) => void;
  onDrawMode: (value: DrawMode) => void;
  onLoadImage: (file: File) => void;
  onClearImage: () => void;
  onToggleFace: (index: number) => void;
  onAddManualFacePath: (path: number[][]) => void;
}

const minimumSize = 16;

export function RegionCanvas({
  mode,
  activeLayer,
  sourceUrl,
  sourceName,
  resultUrl,
  resultName,
  regions,
  selectedId,
  drawMode,
  comparePosition,
  canvasWidth,
  canvasHeight,
  faces,
  selectedFaceIndices,
  manualFacePaths,
  lassoMode,
  onComparePosition,
  onSelect,
  onRegions,
  onDrawMode,
  onLoadImage,
  onClearImage,
  onToggleFace,
  onAddManualFacePath,
}: Props) {
  const stageRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [drag, setDrag] = useState<DragState | null>(null);
  const [draftLasso, setDraftLasso] = useState<number[][]>([]);
  const [frameSize, setFrameSize] = useState({ width: 0, height: 0 });
  const lassoPoints = useRef<number[][]>([]);
  const downloadUrl = resultUrl || sourceUrl;
  const downloadName = resultUrl
    ? (resultName || "k2lab-generated-image.png")
    : (sourceName || "k2lab-canvas-image.png");

  const visibleRegions = mode === "face"
    ? []
    : regions.filter((region) => region.layer === activeLayer);
  const orderedRegions = [
    ...visibleRegions.filter((region) => region.id !== selectedId),
    ...visibleRegions.filter((region) => region.id === selectedId),
  ];

  useLayoutEffect(() => {
    const stage = stageRef.current;
    if (!stage) return undefined;
    function fit(availableWidth: number, availableHeight: number) {
      if (availableWidth <= 0 || availableHeight <= 0 || canvasWidth <= 0 || canvasHeight <= 0) return;
      const scale = Math.min(availableWidth / canvasWidth, availableHeight / canvasHeight);
      const next = { width: canvasWidth * scale, height: canvasHeight * scale };
      setFrameSize((current) => (
        Math.abs(current.width - next.width) < 0.5 && Math.abs(current.height - next.height) < 0.5
          ? current
          : next
      ));
    }
    const style = window.getComputedStyle(stage);
    fit(
      stage.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight),
      stage.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom),
    );
    const observer = new ResizeObserver(([entry]) => {
      if (entry) fit(entry.contentRect.width, entry.contentRect.height);
    });
    observer.observe(stage);
    return () => observer.disconnect();
  }, [canvasWidth, canvasHeight]);

  function point(event: React.PointerEvent<SVGSVGElement | SVGElement>) {
    const rect = svgRef.current!.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(canvasWidth, (event.clientX - rect.left) / rect.width * canvasWidth)),
      y: Math.max(0, Math.min(canvasHeight, (event.clientY - rect.top) / rect.height * canvasHeight)),
    };
  }

  function beginDraw(event: React.PointerEvent<SVGSVGElement>) {
    if (mode === "face") {
      if (!lassoMode || event.target !== event.currentTarget) return;
      const start = point(event);
      event.currentTarget.setPointerCapture(event.pointerId);
      lassoPoints.current = [[start.x, start.y]];
      setDraftLasso(lassoPoints.current);
      setDrag({ kind: "lasso", startX: start.x, startY: start.y });
      return;
    }
    if (!drawMode || event.target !== event.currentTarget) return;
    const start = point(event);
    const names = new Set(
      regions.filter((item) => item.layer === activeLayer).map((item) => item.name.toLocaleLowerCase()),
    );
    const namePrefix = drawMode === "subject" ? "subject" : "region";
    let nameIndex = regions.filter((item) => item.layer === activeLayer).length + 1;
    while (names.has(`${namePrefix} ${nameIndex}`)) nameIndex += 1;
    event.currentTarget.setPointerCapture(event.pointerId);
    const isSubject = drawMode === "subject";
    const region: RegionBox = {
      id: crypto.randomUUID(),
      name: `${isSubject ? "Subject" : "Region"} ${nameIndex}`,
      layer: activeLayer,
      x: start.x,
      y: start.y,
      width: 1,
      height: 1,
      prompt: "",
      faceIdentityPrompt: "",
      spatialRole: isSubject ? "subject" : "auto",
      regionType: isSubject ? "subject" : "region",
      pose: isSubject ? standingPose() : null,
      enabled: true,
    };
    onRegions([...regions, region]);
    onSelect(region.id);
    setDrag({ kind: "draw", regionId: region.id, startX: start.x, startY: start.y });
  }

  function beginMove(event: React.PointerEvent<SVGRectElement>, region: RegionBox) {
    if (drawMode || selectedId !== region.id) return;
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    const start = point(event);
    setDrag({ kind: "move", regionId: region.id, startX: start.x, startY: start.y, initial: region });
  }

  function beginResize(event: React.PointerEvent<SVGRectElement>, region: RegionBox, edge: ResizeEdge) {
    if (drawMode || selectedId !== region.id) return;
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    const start = point(event);
    setDrag({ kind: "resize", regionId: region.id, edge, startX: start.x, startY: start.y, initial: region });
  }

  function beginJoint(
    event: React.PointerEvent<SVGCircleElement>,
    region: RegionBox,
    jointName: PoseJointName,
  ) {
    if (drawMode || selectedId !== region.id || region.regionType !== "subject") return;
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    const start = point(event);
    setDrag({
      kind: "joint",
      regionId: region.id,
      jointName,
      startX: start.x,
      startY: start.y,
      initial: region,
    });
  }

  function beginHead(
    event: React.PointerEvent<SVGElement>,
    region: RegionBox,
    kind: "head-move" | "head-resize-x" | "head-resize-y",
  ) {
    if (drawMode || selectedId !== region.id || !region.pose) return;
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    const start = point(event);
    setDrag({ kind, regionId: region.id, startX: start.x, startY: start.y, initial: region });
  }

  function beginPoseGroup(
    event: React.PointerEvent<SVGElement>,
    region: RegionBox,
    poseGroup: PoseLimbName | "torso",
    kind: "pose-group-move" | "torso-rotate" = "pose-group-move",
  ) {
    if (drawMode || selectedId !== region.id || !region.pose) return;
    event.stopPropagation();
    event.currentTarget.setPointerCapture(event.pointerId);
    const start = point(event);
    setDrag({
      kind,
      regionId: region.id,
      poseGroup,
      startX: start.x,
      startY: start.y,
      initial: region,
    });
  }

  function movePointer(event: React.PointerEvent<SVGSVGElement>) {
    if (!drag) return;
    const current = point(event);
    if (drag.kind === "lasso") {
      const previous = lassoPoints.current.at(-1);
      if (!previous || Math.hypot(current.x - previous[0], current.y - previous[1]) >= 3) {
        lassoPoints.current = [...lassoPoints.current, [current.x, current.y]];
        setDraftLasso(lassoPoints.current);
      }
      return;
    }
    if (!drag.regionId) return;
    onRegions(regions.map((region) => {
      if (region.id !== drag.regionId) return region;
      if (
        region.pose
        && drag.initial?.pose
        && drag.kind === "pose-group-move"
        && drag.poseGroup
      ) {
        const names = drag.poseGroup === "torso"
          ? POSE_TORSO_JOINTS
          : POSE_LIMB_GROUPS[drag.poseGroup];
        return {
          ...region,
          pose: translatePoseGroup(
            drag.initial.pose,
            names,
            (current.x - drag.startX) / region.width,
            (current.y - drag.startY) / region.height,
            drag.poseGroup === "torso",
          ),
        };
      }
      if (
        region.pose
        && drag.initial?.pose
        && drag.kind === "torso-rotate"
      ) {
        const center = poseGroupCenter(drag.initial.pose, [
          "left_shoulder", "right_shoulder", "left_hip", "right_hip",
        ]);
        const centerX = region.x + center.x * region.width;
        const centerY = region.y + center.y * region.height;
        const startAngle = Math.atan2(drag.startY - centerY, drag.startX - centerX);
        const currentAngle = Math.atan2(current.y - centerY, current.x - centerX);
        return {
          ...region,
          pose: rotatePoseGroup(
            drag.initial.pose,
            POSE_JOINT_NAMES,
            center,
            currentAngle - startAngle,
            region.width,
            region.height,
            true,
          ),
        };
      }
      if (drag.kind === "joint" && drag.jointName && region.pose) {
        const x = Math.max(-2, Math.min(3, (current.x - region.x) / region.width));
        const y = Math.max(-2, Math.min(3, (current.y - region.y) / region.height));
        return {
          ...region,
          pose: {
            ...region.pose,
            joints: region.pose.joints.map((joint) => (
              joint.name === drag.jointName ? { ...joint, x, y } : joint
            )),
          },
        };
      }
      if (
        region.pose
        && (drag.kind === "head-move" || drag.kind === "head-resize-x" || drag.kind === "head-resize-y")
      ) {
        const head = region.pose.head;
        const cx = region.x + head.cx * region.width;
        const cy = region.y + head.cy * region.height;
        if (drag.kind === "head-move") {
          return {
            ...region,
            pose: {
              ...region.pose,
              head: {
                ...head,
                cx: Math.max(-2, Math.min(3, (current.x - region.x) / region.width)),
                cy: Math.max(-2, Math.min(3, (current.y - region.y) / region.height)),
              },
            },
          };
        }
        return {
          ...region,
          pose: {
            ...region.pose,
            head: {
              ...head,
              rx: drag.kind === "head-resize-x"
                ? Math.max(0.005, Math.min(1.5, Math.abs(current.x - cx) / region.width))
                : head.rx,
              ry: drag.kind === "head-resize-y"
                ? Math.max(0.005, Math.min(1.5, Math.abs(current.y - cy) / region.height))
                : head.ry,
            },
          },
        };
      }
      if (drag.kind === "draw") {
        return {
          ...region,
          x: Math.min(drag.startX, current.x),
          y: Math.min(drag.startY, current.y),
          width: Math.abs(current.x - drag.startX),
          height: Math.abs(current.y - drag.startY),
        };
      }
      const initial = drag.initial!;
      const dx = current.x - drag.startX;
      const dy = current.y - drag.startY;
      if (drag.kind === "move") {
        return {
          ...region,
          x: Math.max(0, Math.min(canvasWidth - initial.width, initial.x + dx)),
          y: Math.max(0, Math.min(canvasHeight - initial.height, initial.y + dy)),
        };
      }
      return resized(initial, drag.edge!, dx, dy, canvasWidth, canvasHeight);
    }));
  }

  function endPointer() {
    if (drag?.kind === "lasso") {
      if (lassoPoints.current.length >= 3) onAddManualFacePath(lassoPoints.current);
      lassoPoints.current = [];
      setDraftLasso([]);
    }
    if (drag?.kind === "draw" && drag.regionId) {
      const region = regions.find((item) => item.id === drag.regionId);
      if (region && (region.width < minimumSize || region.height < minimumSize)) {
        onRegions(regions.filter((item) => item.id !== region.id));
        onSelect(null);
      }
      onDrawMode(null);
    }
    setDrag(null);
  }

  return (
    <div className="canvas-column">
      <div className="canvas-toolbar">
        <div className="canvas-title">
          <span className="status-dot online" />
          <span>{mode === "edit" ? (activeLayer === "reference" ? "Reference layout" : "Edit targets") : mode === "face" ? "Face refinement source" : "Generation canvas"}</span>
          <small>{sourceName || "1024 × 1024"}</small>
        </div>
        <div className="canvas-actions">
          <label className="quiet-button file-button">
            <Icon name="upload" /> {sourceUrl ? "Replace image" : "Load image"}
            <input type="file" accept={mode === "face" ? "image/png" : "image/png,image/jpeg,image/webp"} onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) onLoadImage(file);
              event.target.value = "";
            }} />
          </label>
          {downloadUrl && (
            <a className="quiet-button canvas-download" href={downloadUrl} download={downloadName}>
              <Icon name="download" /> Download image
            </a>
          )}
          {(sourceUrl || resultUrl) && <button className="quiet-button" onClick={onClearImage}><Icon name="trash" /> Clear canvas</button>}
          {mode !== "face" && (
            <>
              <button
                className={`quiet-button ${drawMode === "region" ? "active" : ""}`}
                onClick={() => onDrawMode(drawMode === "region" ? null : "region")}
              >
                <Icon name="plus" /> {drawMode === "region" ? "Drawing region…" : "Draw region"}
              </button>
              {mode === "generation" && activeLayer === "generation" && (
                <button
                  className={`quiet-button ${drawMode === "subject" ? "active" : ""}`}
                  onClick={() => onDrawMode(drawMode === "subject" ? null : "subject")}
                >
                  <Icon name="face" /> {drawMode === "subject" ? "Drawing subject…" : "Draw subject"}
                </button>
              )}
            </>
          )}
        </div>
      </div>
      <div ref={stageRef} className={`image-stage ${drawMode || lassoMode ? "drawing" : ""}`}>
        <div
          className="image-frame"
          style={{
            width: frameSize.width || undefined,
            height: frameSize.height || undefined,
            aspectRatio: `${canvasWidth} / ${canvasHeight}`,
          }}
        >
          {sourceUrl ? (
            <img className="canvas-image" src={sourceUrl} alt="Loaded source" draggable={false} />
          ) : resultUrl ? (
            <RetryingResultImage source={resultUrl} alt="Generated result" />
          ) : (
            <div className="empty-canvas">
              <div className="empty-orbit"><Icon name={mode === "edit" ? "edit" : mode === "face" ? "face" : "spark"} /></div>
              <strong>{mode === "generation" ? "Start from an open canvas" : "Load an image to begin"}</strong>
              <span>{mode === "generation" ? "Draw regions or add a reference image" : "PNG, JPEG, or WebP"}</span>
            </div>
          )}
          {sourceUrl && resultUrl && (
            <div className="result-clip" style={{ clipPath: `inset(0 ${100 - comparePosition * 100}% 0 0)` }}>
              <RetryingResultImage source={resultUrl} alt="Generation result" />
            </div>
          )}
          <svg
            ref={svgRef}
            className="region-overlay"
            viewBox={`0 0 ${canvasWidth} ${canvasHeight}`}
            preserveAspectRatio="none"
            onPointerDown={beginDraw}
            onPointerMove={movePointer}
            onPointerUp={endPointer}
            onPointerCancel={endPointer}
            onClick={(event) => { if (!drawMode && event.target === event.currentTarget) onSelect(null); }}
          >
            {orderedRegions.map((region) => (
              <g className={`region-group ${region.regionType} ${region.id === selectedId ? "selected" : ""} ${!region.enabled ? "disabled" : ""}`} key={region.id}>
                <rect className="region-fill" x={region.x} y={region.y} width={region.width} height={region.height}
                  onPointerDown={region.id === selectedId ? (event) => beginMove(event, region) : undefined} />
                <rect className="region-outline" x={region.x} y={region.y} width={region.width} height={region.height} />
                <g className="region-label" transform={`translate(${region.x}, ${Math.max(0, region.y - 32)})`}>
                  <rect width={Math.max(112, region.name.length * 13 + 24)} height="28" rx="8" />
                  <text x="12" y="19">{region.regionType === "subject" ? "● " : ""}{region.name}</text>
                </g>
                {region.regionType === "subject" && region.pose?.enabled && (
                  <g className="pose-mannequin">
                    {(() => {
                      const neck = region.pose!.joints.find((joint) => joint.name === "neck")!;
                      return (
                        <line
                          className="pose-volume pose-neck"
                          x1={region.x + region.pose!.head.cx * region.width}
                          y1={region.y + region.pose!.head.cy * region.height}
                          x2={region.x + neck.x * region.width}
                          y2={region.y + neck.y * region.height}
                          strokeWidth={Math.max(8, Math.min(region.width, region.height) * 0.045)}
                        />
                      );
                    })()}
                    {POSE_CONNECTIONS.filter(([from, to]) => !(
                      (from === "left_shoulder" && to === "left_hip")
                      || (from === "right_shoulder" && to === "right_hip")
                      || (from === "left_hip" && to === "right_hip")
                    )).map(([from, to]) => {
                      const first = region.pose!.joints.find((joint) => joint.name === from);
                      const second = region.pose!.joints.find((joint) => joint.name === to);
                      if (!first || !second) return null;
                      const lowerLimb = from.includes("hip") || from.includes("knee");
                      const width = Math.max(8, Math.min(region.width, region.height) * (lowerLimb ? 0.07 : 0.05));
                      return (
                        <line
                          className="pose-volume"
                          key={`${from}-${to}`}
                          x1={region.x + first.x * region.width}
                          y1={region.y + first.y * region.height}
                          x2={region.x + second.x * region.width}
                          y2={region.y + second.y * region.height}
                          strokeWidth={width}
                        />
                      );
                    })}
                    <polygon
                      className="pose-volume pose-torso"
                      points={[
                        "left_shoulder", "right_shoulder", "right_hip", "left_hip",
                      ].map((name) => {
                        const joint = region.pose!.joints.find((item) => item.name === name)!;
                        return `${region.x + joint.x * region.width},${region.y + joint.y * region.height}`;
                      }).join(" ")}
                    />
                    <ellipse
                      className="pose-volume pose-head"
                      cx={region.x + region.pose.head.cx * region.width}
                      cy={region.y + region.pose.head.cy * region.height}
                      rx={region.pose.head.rx * region.width}
                      ry={region.pose.head.ry * region.height}
                    />
                    {region.id === selectedId && region.pose.joints.map((joint) => (
                      <circle
                        key={joint.name}
                        className={region.id === selectedId ? "pose-joint editable" : "pose-joint"}
                        cx={region.x + joint.x * region.width}
                        cy={region.y + joint.y * region.height}
                        r={region.id === selectedId ? 8 : 5}
                        onPointerDown={
                          region.id === selectedId
                            ? (event) => beginJoint(event, region, joint.name)
                            : undefined
                        }
                      />
                    ))}
                    {region.id === selectedId && <>
                      {(() => {
                        const torsoCenter = poseGroupCenter(region.pose!, [
                          "left_shoulder", "right_shoulder", "left_hip", "right_hip",
                        ]);
                        const shoulderCenter = poseGroupCenter(region.pose!, [
                          "left_shoulder", "right_shoulder",
                        ]);
                        const hipCenter = poseGroupCenter(region.pose!, [
                          "left_hip", "right_hip",
                        ]);
                        const directionX = (shoulderCenter.x - hipCenter.x) * region.width;
                        const directionY = (shoulderCenter.y - hipCenter.y) * region.height;
                        const directionLength = Math.max(1, Math.hypot(directionX, directionY));
                        const rotateX = region.x + shoulderCenter.x * region.width + directionX / directionLength * 34;
                        const rotateY = region.y + shoulderCenter.y * region.height + directionY / directionLength * 34;
                        const centerX = region.x + torsoCenter.x * region.width;
                        const centerY = region.y + torsoCenter.y * region.height;
                        return <>
                          <line className="pose-control-leader" x1={centerX} y1={centerY} x2={rotateX} y2={rotateY} />
                          <circle
                            className="pose-group-handle pose-torso-move"
                            cx={centerX}
                            cy={centerY}
                            r={11}
                            onPointerDown={(event) => beginPoseGroup(event, region, "torso")}
                          >
                            <title>Move torso and attached head as one unit</title>
                          </circle>
                          <circle
                            className="pose-rotate-handle"
                            cx={rotateX}
                            cy={rotateY}
                            r={9}
                            onPointerDown={(event) => beginPoseGroup(event, region, "torso", "torso-rotate")}
                          >
                            <title>Rotate the entire figure around the torso center</title>
                          </circle>
                        </>;
                      })()}
                      {(Object.entries(POSE_LIMB_GROUPS) as [PoseLimbName, readonly PoseJointName[]][]).map(([limb, names]) => {
                        const center = poseGroupCenter(region.pose!, names);
                        const side = limb.startsWith("left_") ? -1 : 1;
                        const cx = region.x + center.x * region.width + side * 19;
                        const cy = region.y + center.y * region.height;
                        return (
                          <rect
                            key={limb}
                            className="pose-group-handle pose-limb-move"
                            x={cx - 8}
                            y={cy - 8}
                            width={16}
                            height={16}
                            rx={4}
                            transform={`rotate(45 ${cx} ${cy})`}
                            onPointerDown={(event) => beginPoseGroup(event, region, limb)}
                          >
                            <title>{`Move ${limb.replace("_", " ")} as one unit`}</title>
                          </rect>
                        );
                      })}
                      <circle
                        className="pose-head-handle pose-head-move"
                        cx={region.x + region.pose.head.cx * region.width}
                        cy={region.y + region.pose.head.cy * region.height}
                        r={8}
                        onPointerDown={(event) => beginHead(event, region, "head-move")}
                      />
                      <circle
                        className="pose-head-handle pose-head-resize"
                        cx={region.x + (region.pose.head.cx + region.pose.head.rx) * region.width}
                        cy={region.y + region.pose.head.cy * region.height}
                        r={7}
                        onPointerDown={(event) => beginHead(event, region, "head-resize-x")}
                      />
                      <circle
                        className="pose-head-handle pose-head-resize"
                        cx={region.x + region.pose.head.cx * region.width}
                        cy={region.y + (region.pose.head.cy + region.pose.head.ry) * region.height}
                        r={7}
                        onPointerDown={(event) => beginHead(event, region, "head-resize-y")}
                      />
                    </>}
                  </g>
                )}
                {region.id === selectedId && resizeHandles(region).map((handle) => (
                  <rect key={handle.edge} className={`resize-handle edge-${handle.edge}`}
                    x={handle.x} y={handle.y} width={handle.width} height={handle.height} rx="4"
                    onPointerDown={(event) => beginResize(event, region, handle.edge)} />
                ))}
              </g>
            ))}
            {mode === "face" && manualFacePaths.map((path, index) => (
              <polygon className="manual-face-path" key={`lasso-${index}`} points={path.map((item) => item.join(",")).join(" ")} />
            ))}
            {mode === "face" && draftLasso.length > 0 && (
              <polyline
                className="manual-face-path manual-face-path-draft"
                points={draftLasso.map((item) => item.join(",")).join(" ")}
              />
            )}
            {mode === "face" && faces.map((face) => {
              const [x0, y0, x1, y1] = face.box;
              const selected = selectedFaceIndices.includes(face.index);
              return <g className={`detected-face ${selected ? "selected" : ""}`} key={face.index} onClick={() => onToggleFace(face.index)}>
                <rect x={x0} y={y0} width={x1 - x0} height={y1 - y0} />
                <circle cx={x0 + 15} cy={y0 + 15} r="15" />
                <text x={x0 + 15} y={y0 + 21} textAnchor="middle">{face.index + 1}</text>
              </g>;
            })}
          </svg>
          {sourceUrl && resultUrl && comparePosition > 0 && comparePosition < 1 && (
            <div className="compare-line" style={{ left: `${comparePosition * 100}%` }} />
          )}
          {sourceUrl && resultUrl && (
            <div className="compare-control">
              <span>Source</span>
              <input type="range" min="0" max="1" step="0.01" value={comparePosition}
                onChange={(event) => onComparePosition(Number(event.target.value))} />
              <span>Result</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function RetryingResultImage({ source, alt }: { source: string; alt: string }) {
  const [displaySource, setDisplaySource] = useState(source);
  const retryCount = useRef(0);
  const retryTimer = useRef<number | null>(null);

  useEffect(() => {
    retryCount.current = 0;
    setDisplaySource(source);
    return () => {
      if (retryTimer.current !== null) window.clearTimeout(retryTimer.current);
      retryTimer.current = null;
    };
  }, [source]);

  function retry() {
    if (retryTimer.current !== null || retryCount.current >= 6) return;
    const delay = Math.min(8_000, 500 * (2 ** retryCount.current));
    retryTimer.current = window.setTimeout(() => {
      retryCount.current += 1;
      retryTimer.current = null;
      const separator = source.includes("?") ? "&" : "?";
      setDisplaySource(`${source}${separator}retry=${Date.now()}`);
    }, delay);
  }

  return (
    <img
      className="canvas-image result-image"
      src={displaySource}
      alt={alt}
      draggable={false}
      onError={retry}
      onLoad={() => { retryCount.current = 0; }}
    />
  );
}

function resized(region: RegionBox, edge: ResizeEdge, dx: number, dy: number, canvasWidth: number, canvasHeight: number): RegionBox {
  let left = region.x;
  let top = region.y;
  let right = region.x + region.width;
  let bottom = region.y + region.height;
  if (edge.includes("w")) left = Math.max(0, Math.min(right - minimumSize, left + dx));
  if (edge.includes("e")) right = Math.min(canvasWidth, Math.max(left + minimumSize, right + dx));
  if (edge.includes("n")) top = Math.max(0, Math.min(bottom - minimumSize, top + dy));
  if (edge.includes("s")) bottom = Math.min(canvasHeight, Math.max(top + minimumSize, bottom + dy));
  return { ...region, x: left, y: top, width: right - left, height: bottom - top };
}

function resizeHandles(region: RegionBox) {
  const size = 18;
  const half = size / 2;
  const edge = 12;
  return [
    { edge: "nw" as const, x: region.x - half, y: region.y - half, width: size, height: size },
    { edge: "n" as const, x: region.x + edge, y: region.y - half, width: Math.max(1, region.width - edge * 2), height: size },
    { edge: "ne" as const, x: region.x + region.width - half, y: region.y - half, width: size, height: size },
    { edge: "e" as const, x: region.x + region.width - half, y: region.y + edge, width: size, height: Math.max(1, region.height - edge * 2) },
    { edge: "se" as const, x: region.x + region.width - half, y: region.y + region.height - half, width: size, height: size },
    { edge: "s" as const, x: region.x + edge, y: region.y + region.height - half, width: Math.max(1, region.width - edge * 2), height: size },
    { edge: "sw" as const, x: region.x - half, y: region.y + region.height - half, width: size, height: size },
    { edge: "w" as const, x: region.x - half, y: region.y + edge, width: size, height: Math.max(1, region.height - edge * 2) },
  ];
}
