export const POSE_JOINT_NAMES = [
  "nose",
  "neck",
  "right_shoulder",
  "right_elbow",
  "right_wrist",
  "left_shoulder",
  "left_elbow",
  "left_wrist",
  "right_hip",
  "right_knee",
  "right_ankle",
  "left_hip",
  "left_knee",
  "left_ankle",
  "right_eye",
  "left_eye",
  "right_ear",
  "left_ear",
] as const;

export type PoseJointName = typeof POSE_JOINT_NAMES[number];

export interface PoseJointState {
  name: PoseJointName;
  x: number;
  y: number;
  enabled: boolean;
}

export interface SubjectPoseState {
  enabled: boolean;
  joints: PoseJointState[];
}

export const POSE_CONNECTIONS: [PoseJointName, PoseJointName][] = [
  ["neck", "right_shoulder"],
  ["right_shoulder", "right_elbow"],
  ["right_elbow", "right_wrist"],
  ["neck", "left_shoulder"],
  ["left_shoulder", "left_elbow"],
  ["left_elbow", "left_wrist"],
  ["neck", "right_hip"],
  ["right_hip", "right_knee"],
  ["right_knee", "right_ankle"],
  ["neck", "left_hip"],
  ["left_hip", "left_knee"],
  ["left_knee", "left_ankle"],
  ["neck", "nose"],
  ["nose", "right_eye"],
  ["right_eye", "right_ear"],
  ["nose", "left_eye"],
  ["left_eye", "left_ear"],
];

const standingPoints: Record<PoseJointName, [number, number]> = {
  nose: [0.50, 0.09],
  neck: [0.50, 0.20],
  right_shoulder: [0.36, 0.23],
  right_elbow: [0.29, 0.40],
  right_wrist: [0.28, 0.57],
  left_shoulder: [0.64, 0.23],
  left_elbow: [0.71, 0.40],
  left_wrist: [0.72, 0.57],
  right_hip: [0.43, 0.51],
  right_knee: [0.41, 0.72],
  right_ankle: [0.39, 0.94],
  left_hip: [0.57, 0.51],
  left_knee: [0.59, 0.72],
  left_ankle: [0.61, 0.94],
  right_eye: [0.46, 0.075],
  left_eye: [0.54, 0.075],
  right_ear: [0.42, 0.09],
  left_ear: [0.58, 0.09],
};

const squattingPoints: Record<PoseJointName, [number, number]> = {
  nose: [0.50, 0.25],
  neck: [0.50, 0.36],
  right_shoulder: [0.37, 0.38],
  right_elbow: [0.31, 0.51],
  right_wrist: [0.24, 0.61],
  left_shoulder: [0.63, 0.38],
  left_elbow: [0.69, 0.51],
  left_wrist: [0.76, 0.61],
  right_hip: [0.44, 0.60],
  right_knee: [0.29, 0.71],
  right_ankle: [0.21, 0.92],
  left_hip: [0.56, 0.60],
  left_knee: [0.71, 0.71],
  left_ankle: [0.79, 0.92],
  right_eye: [0.46, 0.235],
  left_eye: [0.54, 0.235],
  right_ear: [0.42, 0.25],
  left_ear: [0.58, 0.25],
};

function poseFromPoints(points: Record<PoseJointName, [number, number]>): SubjectPoseState {
  return {
    enabled: true,
    joints: POSE_JOINT_NAMES.map((name) => ({
      name,
      x: points[name][0],
      y: points[name][1],
      enabled: true,
    })),
  };
}

export function standingPose(): SubjectPoseState {
  return poseFromPoints(standingPoints);
}

export function squattingPose(): SubjectPoseState {
  return poseFromPoints(squattingPoints);
}

export function mirrorPose(pose: SubjectPoseState): SubjectPoseState {
  const mirroredName = (name: PoseJointName): PoseJointName => {
    if (name.startsWith("left_")) return name.replace("left_", "right_") as PoseJointName;
    if (name.startsWith("right_")) return name.replace("right_", "left_") as PoseJointName;
    return name;
  };
  const byName = new Map(
    pose.joints.map((joint) => [
      mirroredName(joint.name),
      { ...joint, name: mirroredName(joint.name), x: 1 - joint.x },
    ]),
  );
  return {
    ...pose,
    joints: POSE_JOINT_NAMES.map((name) => byName.get(name) ?? {
      name,
      x: standingPoints[name][0],
      y: standingPoints[name][1],
      enabled: true,
    }),
  };
}

export function poseFromDocument(value: unknown): SubjectPoseState {
  const fallback = standingPose();
  if (!value || typeof value !== "object" || Array.isArray(value)) return fallback;
  const record = value as Record<string, unknown>;
  if (!Array.isArray(record.joints)) return fallback;
  const byName = new Map<PoseJointName, PoseJointState>();
  record.joints.forEach((item) => {
    if (!item || typeof item !== "object" || Array.isArray(item)) return;
    const joint = item as Record<string, unknown>;
    const name = joint.name;
    if (typeof name !== "string" || !POSE_JOINT_NAMES.includes(name as PoseJointName)) return;
    byName.set(name as PoseJointName, {
      name: name as PoseJointName,
      x: typeof joint.x === "number" ? joint.x : 0.5,
      y: typeof joint.y === "number" ? joint.y : 0.5,
      enabled: joint.enabled !== false,
    });
  });
  return {
    enabled: record.enabled !== false,
    joints: fallback.joints.map((joint) => byName.get(joint.name) ?? joint),
  };
}
