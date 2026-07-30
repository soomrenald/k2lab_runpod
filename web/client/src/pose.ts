export const VOLUMETRIC_POSE_FORMAT = "k2-volumetric-pose-v1" as const;

export const POSE_JOINT_NAMES = [
  "neck",
  "left_shoulder",
  "right_shoulder",
  "left_elbow",
  "right_elbow",
  "left_wrist",
  "right_wrist",
  "left_hip",
  "right_hip",
  "left_knee",
  "right_knee",
  "left_ankle",
  "right_ankle",
] as const;

export type PoseJointName = typeof POSE_JOINT_NAMES[number];

export const POSE_TORSO_JOINTS: PoseJointName[] = [
  "neck",
  "left_shoulder",
  "right_shoulder",
  "left_hip",
  "right_hip",
];

export const POSE_LIMB_GROUPS = {
  left_arm: ["left_shoulder", "left_elbow", "left_wrist"],
  right_arm: ["right_shoulder", "right_elbow", "right_wrist"],
  left_leg: ["left_hip", "left_knee", "left_ankle"],
  right_leg: ["right_hip", "right_knee", "right_ankle"],
} as const satisfies Record<string, readonly PoseJointName[]>;

export type PoseLimbName = keyof typeof POSE_LIMB_GROUPS;

export interface PoseJointState {
  name: PoseJointName;
  x: number;
  y: number;
}

export interface PoseHeadState {
  cx: number;
  cy: number;
  rx: number;
  ry: number;
}

export interface SubjectPoseState {
  format: typeof VOLUMETRIC_POSE_FORMAT;
  enabled: boolean;
  joints: PoseJointState[];
  head: PoseHeadState;
}

export function poseGroupCenter(
  pose: SubjectPoseState,
  names: readonly PoseJointName[],
): { x: number; y: number } {
  const selected = new Set<PoseJointName>(names);
  const joints = pose.joints.filter((joint) => selected.has(joint.name));
  if (joints.length === 0) return { x: 0.5, y: 0.5 };
  return {
    x: joints.reduce((sum, joint) => sum + joint.x, 0) / joints.length,
    y: joints.reduce((sum, joint) => sum + joint.y, 0) / joints.length,
  };
}

export function translatePoseGroup(
  pose: SubjectPoseState,
  names: readonly PoseJointName[],
  dx: number,
  dy: number,
  moveHead = false,
): SubjectPoseState {
  const selected = new Set<PoseJointName>(names);
  const points = pose.joints.filter((joint) => selected.has(joint.name));
  if (moveHead) points.push({ name: "neck", x: pose.head.cx, y: pose.head.cy });
  const boundedDx = Math.max(
    Math.max(...points.map((point) => -2 - point.x)),
    Math.min(Math.min(...points.map((point) => 3 - point.x)), dx),
  );
  const boundedDy = Math.max(
    Math.max(...points.map((point) => -2 - point.y)),
    Math.min(Math.min(...points.map((point) => 3 - point.y)), dy),
  );
  return {
    ...pose,
    joints: pose.joints.map((joint) => (
      selected.has(joint.name)
        ? { ...joint, x: joint.x + boundedDx, y: joint.y + boundedDy }
        : joint
    )),
    head: moveHead
      ? { ...pose.head, cx: pose.head.cx + boundedDx, cy: pose.head.cy + boundedDy }
      : pose.head,
  };
}

export function rotatePoseGroup(
  pose: SubjectPoseState,
  names: readonly PoseJointName[],
  center: { x: number; y: number },
  radians: number,
  scaleX: number,
  scaleY: number,
  moveHead = false,
): SubjectPoseState {
  const selected = new Set<PoseJointName>(names);
  const cosine = Math.cos(radians);
  const sine = Math.sin(radians);
  const rotate = (x: number, y: number) => {
    const offsetX = (x - center.x) * scaleX;
    const offsetY = (y - center.y) * scaleY;
    return {
      x: center.x + (offsetX * cosine - offsetY * sine) / scaleX,
      y: center.y + (offsetX * sine + offsetY * cosine) / scaleY,
    };
  };
  const rotatedHead = rotate(pose.head.cx, pose.head.cy);
  return {
    ...pose,
    joints: pose.joints.map((joint) => (
      selected.has(joint.name) ? { ...joint, ...rotate(joint.x, joint.y) } : joint
    )),
    head: moveHead
      ? { ...pose.head, cx: rotatedHead.x, cy: rotatedHead.y }
      : pose.head,
  };
}

export const POSE_CONNECTIONS: [PoseJointName, PoseJointName][] = [
  ["neck", "left_shoulder"],
  ["left_shoulder", "left_elbow"],
  ["left_elbow", "left_wrist"],
  ["neck", "right_shoulder"],
  ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_wrist"],
  ["left_shoulder", "left_hip"],
  ["right_shoulder", "right_hip"],
  ["left_hip", "right_hip"],
  ["left_hip", "left_knee"],
  ["left_knee", "left_ankle"],
  ["right_hip", "right_knee"],
  ["right_knee", "right_ankle"],
];

const standingPoints: Record<PoseJointName, [number, number]> = {
  neck: [0.50, 0.20],
  left_shoulder: [0.39, 0.24],
  right_shoulder: [0.61, 0.24],
  left_elbow: [0.34, 0.42],
  right_elbow: [0.66, 0.42],
  left_wrist: [0.31, 0.60],
  right_wrist: [0.69, 0.60],
  left_hip: [0.44, 0.52],
  right_hip: [0.56, 0.52],
  left_knee: [0.43, 0.72],
  right_knee: [0.57, 0.72],
  left_ankle: [0.42, 0.94],
  right_ankle: [0.58, 0.94],
};

const squattingPoints: Record<PoseJointName, [number, number]> = {
  neck: [0.50, 0.36],
  left_shoulder: [0.37, 0.38],
  right_shoulder: [0.63, 0.38],
  left_elbow: [0.31, 0.51],
  right_elbow: [0.69, 0.51],
  left_wrist: [0.24, 0.61],
  right_wrist: [0.76, 0.61],
  left_hip: [0.44, 0.60],
  right_hip: [0.56, 0.60],
  left_knee: [0.29, 0.71],
  right_knee: [0.71, 0.71],
  left_ankle: [0.21, 0.92],
  right_ankle: [0.79, 0.92],
};

function poseFromPoints(
  points: Record<PoseJointName, [number, number]>,
  head: PoseHeadState,
): SubjectPoseState {
  return {
    format: VOLUMETRIC_POSE_FORMAT,
    enabled: true,
    joints: POSE_JOINT_NAMES.map((name) => ({
      name,
      x: points[name][0],
      y: points[name][1],
    })),
    head,
  };
}

export function standingPose(): SubjectPoseState {
  return poseFromPoints(standingPoints, { cx: 0.50, cy: 0.105, rx: 0.075, ry: 0.105 });
}

export function squattingPose(): SubjectPoseState {
  return poseFromPoints(squattingPoints, { cx: 0.50, cy: 0.265, rx: 0.075, ry: 0.105 });
}

export function mirrorPose(pose: SubjectPoseState): SubjectPoseState {
  const byName = new Map(pose.joints.map((joint) => [joint.name, joint]));
  const sourceName = (name: PoseJointName): PoseJointName => {
    if (name.startsWith("left_")) return name.replace("left_", "right_") as PoseJointName;
    if (name.startsWith("right_")) return name.replace("right_", "left_") as PoseJointName;
    return name;
  };
  return {
    ...pose,
    joints: POSE_JOINT_NAMES.map((name) => {
      const source = byName.get(sourceName(name));
      return {
        name,
        x: 1 - (source?.x ?? standingPoints[name][0]),
        y: source?.y ?? standingPoints[name][1],
      };
    }),
    head: { ...pose.head, cx: 1 - pose.head.cx },
  };
}

function finite(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function legacyHead(
  values: Map<string, { x: number; y: number }>,
  neck: PoseJointState,
): PoseHeadState {
  const face = ["right_eye", "left_eye", "right_ear", "left_ear"]
    .map((name) => values.get(name))
    .filter((point): point is { x: number; y: number } => Boolean(point));
  const nose = values.get("nose");
  const cx = nose?.x ?? (face.length
    ? face.reduce((sum, point) => sum + point.x, 0) / face.length
    : neck.x);
  const cy = nose?.y ?? (face.length
    ? face.reduce((sum, point) => sum + point.y, 0) / face.length
    : neck.y - 0.095);
  const rx = Math.max(0.04, Math.min(0.20, ...face.map((point) => Math.abs(point.x - cx)), 0.075));
  return { cx, cy, rx, ry: Math.max(0.055, Math.min(0.20, Math.abs(neck.y - cy))) };
}

export function poseFromDocument(value: unknown): SubjectPoseState {
  const fallback = standingPose();
  if (!value || typeof value !== "object" || Array.isArray(value)) return fallback;
  const record = value as Record<string, unknown>;
  const values = new Map<string, { x: number; y: number }>();
  if (Array.isArray(record.joints)) {
    record.joints.forEach((item) => {
      if (!item || typeof item !== "object" || Array.isArray(item)) return;
      const joint = item as Record<string, unknown>;
      if (typeof joint.name !== "string") return;
      values.set(joint.name, {
        x: finite(joint.x, 0.5),
        y: finite(joint.y, 0.5),
      });
    });
  } else if (record.joints && typeof record.joints === "object") {
    Object.entries(record.joints as Record<string, unknown>).forEach(([name, item]) => {
      if (!item || typeof item !== "object" || Array.isArray(item)) return;
      const joint = item as Record<string, unknown>;
      values.set(name, {
        x: finite(joint.x, 0.5),
        y: finite(joint.y, 0.5),
      });
    });
  }
  const joints = fallback.joints.map((joint) => ({
    ...joint,
    ...(values.get(joint.name) ?? {}),
  }));
  const headValue = record.head && typeof record.head === "object" && !Array.isArray(record.head)
    ? record.head as Record<string, unknown>
    : null;
  const head = record.format === VOLUMETRIC_POSE_FORMAT && headValue
    ? {
      cx: finite(headValue.cx, fallback.head.cx),
      cy: finite(headValue.cy, fallback.head.cy),
      rx: finite(headValue.rx, fallback.head.rx),
      ry: finite(headValue.ry, fallback.head.ry),
    }
    : legacyHead(values, joints.find((joint) => joint.name === "neck")!);
  return {
    format: VOLUMETRIC_POSE_FORMAT,
    enabled: record.enabled !== false,
    joints,
    head,
  };
}
