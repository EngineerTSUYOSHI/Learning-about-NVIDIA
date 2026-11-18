"""SO-101 vial placement from full state, with an opt-in ``sim2real`` preset.

Selecting ``presets=sim2real`` enables the physical randomization used for robot transfer. The wrist-camera task in
:mod:`.camera_env_cfg` inherits this task, so a state teacher and its camera student see the same physics.
"""

from __future__ import annotations

import math
from typing import Any

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import RelativeJointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysicsEvent, PhysicsManager, PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.utils import bind_physics_material, clone
from isaaclab.utils.configclass import configclass
from isaaclab.visualizers import VisualizerCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
from isaaclab_newton.physics.newton_manager_cfg import NewtonBuilderCfg
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

from isaaclab_tasks.utils import PresetCfg, preset

from isaaclab_assets.robots import so101

from isaaclab_tutorial.assets import BALL_USD, DESK_USD
from isaaclab_tutorial.tasks.place_vial import mdp

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINTS = JOINTS[:-1]

# ボール直径のランダム化設定 [m]（環境ごとに起動時に割り当てられ、その環境では学習中固定される）
# ランダム化する場合の例: BALL_DIAMETER_RANGE = (0.01, 0.06), BALL_DIAMETER_STEP = 0.005（1cm〜6cmの5mm刻み）
BALL_DIAMETER_RANGE: tuple[float, float] = (0.05, 0.05)  # 今回は 5cm 固定
BALL_DIAMETER_STEP: float = 0.005
# シミュレーション起動時に各環境へ割り当てられたボール半径 [m] を保持するリスト
_ENV_BALL_RADII: list[float] = []

# 直径5cmボールが無理なく爪の間に入るよう、初期開き角度を少し広め（30度）に設定
PREGRASP_GRIPPER_POSITION = math.radians(30.0)
GRASP_GRIPPER_POSITION = math.radians(-8.9)
RELEASE_GRIPPER_POSITION = math.radians(36.97)

WORKSHOP_INITIAL_JOINT_POSITION = (
    -0.1221070742,
    -0.9066845838,
    0.1900876486 + math.radians(6.4),
    1.4797928525,
    -0.8044013083,
    PREGRASP_GRIPPER_POSITION,
)

# 把持状態からスタートする際の基準関節角度（机のすぐ上でボールを挟んでいるポーズ）
WORKSHOP_GRASP_JOINT_POSITION = (
    0.09114071,
    0.44803178,
    -0.43491304 + math.radians(6.4),
    1.15299133,
    -1.64141933,
    math.radians(10.0),
)

# 持ち上げ途中の基準関節角度（把持スタート時の高さバリエーション補間用）
WORKSHOP_LIFT_MID_JOINT_POSITION = (
    -0.03038024,
    0.04142760,
    -0.33287840 + math.radians(6.4),
    1.23753429,
    -1.64005544,
    math.radians(10.0),
)

# The sim2real preset seats the robot base on the 35 mm tabletop, as on the real robot.
BASE_SEATING_Z = 0.030081447775408447
TABLETOP_Z = 0.035
BASE_HEIGHT_CORRECTION = TABLETOP_Z - BASE_SEATING_Z

_CONTACT_STIFFNESS = 1.57e5
_CONTACT_DAMPING = 1.12e3
_FRICTION = 0.7
_ROLLING_FRICTION = 0.05
_TORSIONAL_FRICTION = 0.005
# Sliding-only contacts, so the rolling and torsional coefficients above have no effect.
_CONDIM = 3
_SOLIMP = (0.7, 0.95, 0.0001, 0.5, 2.0)
_SOLREF = (0.002, 1.5)
_contact_model_registered = False


# Bound to every collider so both backends see the same friction. PhysX also uses it as a compliant contact matching
# Newton's soft contact; rigid PhysX contacts let the vial slip out of the jaws during the lift.
WORKSHOP_CONTACT_MATERIAL = PhysxRigidBodyMaterialCfg(
    static_friction=_FRICTION,
    dynamic_friction=_FRICTION,
    restitution=0.0,
    friction_combine_mode="max",
    compliant_contact_stiffness=_CONTACT_STIFFNESS,
    compliant_contact_damping=_CONTACT_DAMPING,
    compliant_contact_acceleration_spring=True,
)


def _bind_workshop_contact_material(prim: Any, prim_path: str) -> None:
    from isaaclab.sim.spawners.materials.physics_materials import spawn_physics_material

    material_path = f"{prim_path}/workshopContactMaterial"
    spawn_physics_material(material_path, WORKSHOP_CONTACT_MATERIAL, stage=prim.GetStage())
    bind_physics_material(prim_path, material_path, stage=prim.GetStage())


@clone
def _spawn_usd_with_contact_material(
    prim_path: str,
    cfg: Any,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn a USD asset and bind the workshop contact material to all of its colliders."""
    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd

    prim = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    _bind_workshop_contact_material(prim, prim_path)
    return prim


# --- STEP 1: ボールのスポーン設定（assets/workshop/ball.usda を読み込む） ---
# ※補足: ボールのような単純な球体であれば、外部のUSDファイルを用意しなくても
# sim_utils.SphereCfg(radius=..., rigid_props=..., collision_props=...) を使って
# コード上から直接プリミティブ形状として生成する方法もあります。
@configclass
class BallSpawnCfg(sim_utils.UsdFileCfg):
    """assets/workshop/ball.usda を読み込み、接触マテリアルをバインドする設定。"""

    func = _spawn_usd_with_contact_material


@clone
def _spawn_open_box(
    prim_path: str,
    cfg: OpenBoxSpawnCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """プリミティブ（立方体5枚）を組み合わせて上開きの箱（カゴ）を生成する。"""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    import isaaclab.sim as sim_utils_inner

    stage = sim_utils_inner.get_current_stage()
    root_xform = UsdGeom.Xform.Define(stage, prim_path)
    root_prim = root_xform.GetPrim()
    if translation is not None:
        root_xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    if orientation is not None:
        # Isaac Lab の回転表現は XYZW 順、Gf.Quatf の引数は (w, x, y, z) 順
        root_xform.AddOrientOp().Set(Gf.Quatf(orientation[3], orientation[0], orientation[1], orientation[2]))

    # 静的・キネマティック剛体として定義（ボールが当たっても箱自体は動かない）
    rb_api = UsdPhysics.RigidBodyAPI.Apply(root_prim)
    rb_api.CreateKinematicEnabledAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(root_prim)
    mass_api.CreateMassAttr(1.0)

    # 見た目のマテリアル（落ち着いた緑系の箱）
    looks_path = f"{prim_path}/Looks"
    UsdGeom.Scope.Define(stage, looks_path)
    mat_path = f"{looks_path}/BoxMaterial"
    mat = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*cfg.color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    s = cfg.size
    h = cfg.height
    t = cfg.wall_thickness
    # 底板1枚 + 四方の壁4枚（各Cubeは size=1.0 にして scale で寸法を指定）
    parts = (
        ("floor", (0.0, 0.0, 0.5 * t), (s, s, t)),
        ("wall_px", (0.5 * s - 0.5 * t, 0.0, 0.5 * h), (t, s, h)),
        ("wall_nx", (-0.5 * s + 0.5 * t, 0.0, 0.5 * h), (t, s, h)),
        ("wall_py", (0.0, 0.5 * s - 0.5 * t, 0.5 * h), (s - 2.0 * t, t, h)),
        ("wall_ny", (0.0, -0.5 * s + 0.5 * t, 0.5 * h), (s - 2.0 * t, t, h)),
    )
    for name, pos, scale in parts:
        cube = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
        cube.CreateSizeAttr(1.0)
        cube.AddTranslateOp().Set(Gf.Vec3d(*pos))
        cube.AddScaleOp().Set(Gf.Vec3f(*scale))
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)

    _bind_workshop_contact_material(root_prim, prim_path)
    return root_prim


@configclass
class OpenBoxSpawnCfg(sim_utils.SpawnerCfg):
    """プリミティブ形状で上開きの箱（11cm x 11cm x 高さ5cm）を生成する設定。"""

    func = _spawn_open_box
    size: float = 0.11
    height: float = 0.05
    wall_thickness: float = 0.006
    color: tuple[float, float, float] = (0.15, 0.55, 0.30)


@clone
def _spawn_support(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from pxr import Sdf

    prim = sim_utils.spawn_cuboid(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    prim.CreateAttribute("tutorial:rollingFrictionRange", Sdf.ValueTypeNames.Float2, custom=True).Set(
        cfg.rolling_friction_range
    )
    prim.CreateAttribute("tutorial:torsionalFrictionRange", Sdf.ValueTypeNames.Float2, custom=True).Set(
        cfg.torsional_friction_range
    )
    return prim


@configclass
class SupportSpawnCfg(sim_utils.CuboidCfg):
    """Gray mat under the task objects, with randomized rolling/torsional resistance."""

    func = _spawn_support
    # MuJoCo rolling/torsional coefficients are torque/normal-force lengths [m].
    rolling_friction_range: tuple[float, float] = (0.0002, 0.002)
    torsional_friction_range: tuple[float, float] = (0.001, 0.005)


def _sample_ball_radii(num_worlds: int, diameter_range: tuple[float, float], step: float) -> list[float]:
    """指定範囲・刻み幅から均等にボール半径リスト [m] を生成してシャッフルする。"""
    lo, hi = diameter_range
    if hi <= lo or step <= 0.0:
        return [0.5 * lo] * num_worlds
    count = max(1, int(round((hi - lo) / step)) + 1)
    candidates = [0.5 * min(hi, lo + i * step) for i in range(count)]
    perm = torch.randperm(num_worlds).tolist()
    return [candidates[idx % len(candidates)] for idx in perm]


def _initialize_contacts(_event: PhysicsEvent) -> None:
    """Apply the workshop-validated contact model and per-env ball sizes to every Newton shape."""
    from pxr import Gf, UsdGeom

    sim = sim_utils.SimulationContext.instance()
    builder = sim.get_or_create_backend(NewtonBuilderCfg(physics_cfg=sim.cfg.physics))

    # ボールサイズの環境ごとの割り当て（物理コライダとUSD見た目メッシュの両方を同期スケール）
    num_worlds = max((w for w in builder.shape_world), default=0) + 1
    radii = _sample_ball_radii(num_worlds, BALL_DIAMETER_RANGE, BALL_DIAMETER_STEP)
    _ENV_BALL_RADII.clear()
    _ENV_BALL_RADII.extend(radii)

    base_radius = mdp.BALL_RADIUS  # ball.usda の基準半径 (0.025 m)
    if any(abs(r - base_radius) > 1.0e-6 for r in radii):
        for index, label in enumerate(builder.shape_label):
            if "/Ball/" not in label:
                continue
            world = builder.shape_world[index]
            factor = radii[world] / base_radius
            scale = list(builder.shape_scale[index])
            builder.shape_scale[index] = tuple(s * factor for s in scale)
        # 視覚（USD）側のスケールも各環境ごとに合わせる
        for world, r in enumerate(radii):
            factor = r / base_radius
            geom_prim = sim.stage.GetPrimAtPath(f"/World/envs/env_{world}/Ball/Geometry")
            if geom_prim and geom_prim.IsValid():
                xform = UsdGeom.Xformable(geom_prim)
                scale_ops = [op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeScale]
                scale_op = scale_ops[0] if scale_ops else xform.AddScaleOp()
                scale_op.Set(Gf.Vec3f(factor, factor, factor))

    num_shapes = len(builder.shape_body)
    for shape_index in range(num_shapes):
        builder.shape_material_ke[shape_index] = _CONTACT_STIFFNESS
        builder.shape_material_kd[shape_index] = _CONTACT_DAMPING
        builder.shape_material_mu[shape_index] = _FRICTION
        builder.shape_material_mu_rolling[shape_index] = _ROLLING_FRICTION
        builder.shape_material_mu_torsional[shape_index] = _TORSIONAL_FRICTION

    for name, value in (("mujoco:geom_solimp", _SOLIMP), ("mujoco:geom_solref", _SOLREF), ("mujoco:condim", _CONDIM)):
        attribute = builder.custom_attributes.get(name)
        if attribute is None:
            continue
        if attribute.values is None:
            attribute.values = {}
        for shape_index in range(num_shapes):
            attribute.values[shape_index] = value

    # The vial's cap is wider than its body, so on its side it rolls in circles unless the desk resists rolling.
    rolling = (_ROLLING_FRICTION, _ROLLING_FRICTION)
    torsional = (_TORSIONAL_FRICTION, _TORSIONAL_FRICTION)
    support = sim.stage.GetPrimAtPath("/World/envs/env_0/Support")
    if support:
        support_rolling = support.GetAttribute("tutorial:rollingFrictionRange").Get()
        support_torsional = support.GetAttribute("tutorial:torsionalFrictionRange").Get()
        if support_rolling is not None and support_torsional is not None:
            rolling, torsional = tuple(support_rolling), tuple(support_torsional)
    configure_support_rolling_contacts(builder, rolling, torsional)


def configure_support_rolling_contacts(builder, rolling_range, torsional_range) -> None:
    """Enable rolling resistance only at the desk and mat, sampled per world.

    Pair friction uses the maximum of both shapes, so the rolling coefficients of every other shape are cleared
    before the desk/mat request a six-dimensional contact.
    """
    per_world = {}
    condim = builder.custom_attributes["mujoco:condim"].values
    for index, label in enumerate(builder.shape_label):
        world = builder.shape_world[index]
        if world not in per_world:
            u, v = torch.rand(2).tolist()
            per_world[world] = (
                rolling_range[0] + u * (rolling_range[1] - rolling_range[0]),
                torsional_range[0] + v * (torsional_range[1] - torsional_range[0]),
            )
        surface = "/Support/" in label or "/Desk/" in label
        builder.shape_material_mu_rolling[index] = per_world[world][0] if surface else 0.0
        builder.shape_material_mu_torsional[index] = per_world[world][1] if surface else 0.0
        if surface:
            condim[index] = 6


def _register_contact_model() -> None:
    """Register the contact initializer once per process."""
    global _contact_model_registered
    if _contact_model_registered:
        return
    PhysicsManager.register_callback(
        _initialize_contacts,
        PhysicsEvent.MODEL_INIT,
        name="so101_workshop_contact_model",
    )
    _contact_model_registered = True


_register_contact_model()


def _remove_non_gripper_colliders(robot_prim: Any) -> None:
    """Keep fixed jaw, moving jaw and camera-mount contacts; retain all link dynamics/visuals."""
    from pxr import Usd, UsdPhysics

    root = robot_prim.GetPath()
    for prim in Usd.PrimRange(robot_prim):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        link = prim.GetPath().MakeRelativePath(root).pathString.split("/", 1)[0]
        if link not in ("gripper", "moving_jaw_so101_v1"):
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
            prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
            prim.RemoveProperty("physics:collisionEnabled")
            prim.RemoveProperty("physics:approximation")


@clone
def _spawn_so101(
    prim_path: str,
    cfg: SO101SpawnCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn the workshop SO-101: calibrated joints, contact material, gripper-only colliders, orange plastic."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd

    prim = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    _bind_workshop_contact_material(prim, prim_path)
    stage = prim.GetStage()
    for joint, response in (("shoulder_lift", cfg.shoulder_response), ("elbow_flex", cfg.elbow_response)):
        if response is None:
            continue
        joint_prim = stage.GetPrimAtPath(f"{prim_path}/joints/{joint}")
        for name, factor in zip(
            ("drive:angular:physics:stiffness", "drive:angular:physics:damping", "newton:friction"),
            response,
            strict=True,
        ):
            attribute = joint_prim.GetAttribute(name)
            attribute.Set(attribute.Get() * factor)
        # Stop joints creeping under small commands, as the real follower does.
        joint_prim.CreateAttribute("mjc:solimpfriction", Sdf.ValueTypeNames.FloatArray).Set(
            [0.9999, 0.9999, 0.001, 0.5, 2.0]
        )
    # Workshop inclinometer correction: shift the joint coordinate and frame together.
    elbow = stage.GetPrimAtPath(f"{prim_path}/joints/elbow_flex")
    elbow.GetAttribute("physics:localRot0").Set(Gf.Quatf(0.7454748, Gf.Vec3f(-6.239807e-16, -2.6585315e-16, 0.6665338)))
    elbow.GetAttribute("physics:lowerLimit").Set(-90.42986)
    elbow.GetAttribute("physics:upperLimit").Set(103.22987)
    elbow.GetAttribute("drive:angular:physics:targetPosition").Set(6.4)
    gripper = stage.GetPrimAtPath(f"{prim_path}/joints/gripper")
    # This follower has 1449 calibrated ticks; do not substitute another robot's span.
    gripper.GetAttribute("physics:lowerLimit").Set(-12.505751884817162)
    gripper.GetAttribute("physics:upperLimit").Set(-12.505751884817162 + 1449 * 360 / 4095)
    camera = stage.GetPrimAtPath(f"{prim_path}/gripper/wowrobo_2MP_camera")
    camera.GetAttribute("clippingRange").Set(Gf.Vec2f(0.001, 5.0))
    if cfg.hide_camera_housing:
        # Newton ray tracing ignores the near clipping plane, so the camera would otherwise see its own housing.
        UsdGeom.Imageable(stage.GetPrimAtPath(f"{prim_path}/gripper/visuals/camera_mount")).MakeInvisible()
    # Recolor only printed plastic; preserve the black servos and camera housing.
    material = UsdShade.Material(stage.GetPrimAtPath(f"{prim_path}/Looks/material_a_d_printed"))
    material.GetPrim().SetInstanceable(False)
    shader, _, _ = material.ComputeSurfaceSource("mdl")
    if not shader:
        shader, _, _ = material.ComputeSurfaceSource()
    color = shader.GetInput("diffuse_color_constant") or shader.GetInput("diffuseColor")
    color.Set(Gf.Vec3f(1.0, 0.22, 0.015))
    _remove_non_gripper_colliders(prim)
    return prim


@configclass
class SO101SpawnCfg(sim_utils.UsdFileCfg):
    """SO-101 asset with optional fits of this follower's measured joint response."""

    func = _spawn_so101
    # Multipliers on authored stiffness, drive damping and Coulomb friction.
    shoulder_response: tuple[float, float, float] | None = None
    elbow_response: tuple[float, float, float] | None = None
    hide_camera_housing: bool = False


# System-identified joint dynamics. The asset authors them for Newton only; the PhysX preset applies them through the
# actuator config.
SYS_ID_ARMATURE = {
    "shoulder_pan": 0.06762,
    "shoulder_lift": 0.027645,
    "elbow_flex": 0.03772,
    "wrist_flex": 0.050714,
    "wrist_roll": 0.054898,
    "gripper": 0.077625,
}
SYS_ID_JOINT_DAMPING = {  # N m s / rad
    "shoulder_pan": math.degrees(0.0277631095),
    "shoulder_lift": math.degrees(0.00979871475),
    "elbow_flex": math.degrees(0.0139039735),
    "wrist_flex": math.degrees(0.0187076781),
    "wrist_roll": math.degrees(0.028582751),
    "gripper": math.degrees(0.0167301153),
}
SYS_ID_JOINT_FRICTION = {
    "shoulder_pan": 0.347432,
    "shoulder_lift": 0.344793,
    "elbow_flex": 0.41119,
    "wrist_flex": 0.248233,
    "wrist_roll": 0.221761,
    "gripper": 0.083458,
}

# Multipliers fitted so PhysX joint responses match Newton's.
PHYSX_JOINT_FRICTION_SCALE = {
    "shoulder_pan": 1.5,
    "shoulder_lift": 1.5,
    "elbow_flex": 1.0,
    "wrist_flex": 1.0,
    "wrist_roll": 1.5,
    "gripper": 2.0,
}
PHYSX_JOINT_VISCOUS_SCALE = {
    "shoulder_pan": 2.0,
    "shoulder_lift": 3.0,
    "elbow_flex": 1.5,
    "wrist_flex": 3.0,
    "wrist_roll": 2.0,
    "gripper": 2.0,
}
PHYSX_JOINT_FRICTION = {name: value * PHYSX_JOINT_FRICTION_SCALE[name] for name, value in SYS_ID_JOINT_FRICTION.items()}
PHYSX_JOINT_DAMPING = {name: value * PHYSX_JOINT_VISCOUS_SCALE[name] for name, value in SYS_ID_JOINT_DAMPING.items()}


def _subset(values: dict[str, float], names: list[str]) -> dict[str, float]:
    return {name: values[name] for name in names}


# Halving vial friction on PhysX reproduces Newton's in-hand grasp pose.
VIAL_FRICTION_RANGE = (0.7, 1.3)
PHYSX_VIAL_FRICTION_RANGE = tuple(v * 0.5 for v in VIAL_FRICTION_RANGE)

# A softer gripper drive on PhysX reproduces Newton's stall angle against the vial.
PHYSX_GRIPPER_STIFFNESS = 68.2508 * 0.6

# PhysX under-converges the jaw/vial pinch at its default iteration count.
PHYSX_SOLVER_POSITION_ITERATIONS = 128

SO101_CFG = so101.SO101_CFG.replace(
    spawn=SO101SpawnCfg(
        **{
            **vars(so101.SO101_CFG.spawn),
            "func": _spawn_so101,
            # The asset instances its collision meshes; binding the contact material needs real prims.
            "make_uninstanceable": True,
            "activate_contact_sensors": True,
            "variants": {
                "Robot": "robot",
                "Sensor": "sensors",
                "Physics": preset(default="physics", newton_mjwarp="physics", physx="physx"),
            },
        },
        # Fitted to this follower's measured shoulder/elbow response.
        shoulder_response=preset(default=None, sim2real=(0.22, 0.1, 0.4)),
        elbow_response=preset(default=None, sim2real=(1.0, 1.0, 0.8)),
    ),
    # Separate gripper group so it can carry a PhysX-only drive stiffness.
    actuators={
        "arm": so101.SO101_CFG.actuators["usd"].replace(
            joint_names_expr=ARM_JOINTS,
            armature=preset(default=None, newton_mjwarp=None, physx=_subset(SYS_ID_ARMATURE, ARM_JOINTS)),
            friction=preset(default=None, newton_mjwarp=None, physx=_subset(PHYSX_JOINT_FRICTION, ARM_JOINTS)),
            dynamic_friction=preset(default=None, newton_mjwarp=None, physx=_subset(PHYSX_JOINT_FRICTION, ARM_JOINTS)),
            viscous_friction=preset(default=None, newton_mjwarp=None, physx=_subset(PHYSX_JOINT_DAMPING, ARM_JOINTS)),
        ),
        "gripper": so101.SO101_CFG.actuators["usd"].replace(
            joint_names_expr=["gripper"],
            armature=preset(default=None, newton_mjwarp=None, physx=_subset(SYS_ID_ARMATURE, ["gripper"])),
            friction=preset(default=None, newton_mjwarp=None, physx=_subset(PHYSX_JOINT_FRICTION, ["gripper"])),
            dynamic_friction=preset(default=None, newton_mjwarp=None, physx=_subset(PHYSX_JOINT_FRICTION, ["gripper"])),
            viscous_friction=preset(default=None, newton_mjwarp=None, physx=_subset(PHYSX_JOINT_DAMPING, ["gripper"])),
            stiffness=preset(default=None, newton_mjwarp=None, physx={"gripper": PHYSX_GRIPPER_STIFFNESS}),
        ),
    },
)


def _jaw_support_sensor(link: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{link}",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Desk", "{ENV_REGEX_NS}/Support"],
        history_length=1,
    )


@configclass
class SO101SceneCfg(InteractiveSceneCfg):
    """One SO-101, one vial, one rack, and a desk; sim2real adds the gray mat and jaw/support contact sensing."""

    robot = SO101_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=SO101_CFG.init_state.replace(
            # 背面側の箱も机の上に収まるよう、ロボットを机中央寄り（x=0.10）へ配置
            pos=preset(default=(0.10, 0.0, 0.0), sim2real=(0.10, 0.0, BASE_HEIGHT_CORRECTION)),
            # +90 degrees about world Z (XYZW).
            rot=(0.0, 0.0, 0.7071068, 0.7071068),
            joint_pos=dict(zip(JOINTS, WORKSHOP_INITIAL_JOINT_POSITION, strict=True)),
        ),
        soft_joint_pos_limit_factor=0.98,
    )

    # 直径5cmのボールオブジェクト（assets/workshop/ball.usda から読み込み）
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=BallSpawnCfg(
            usd_path=str(BALL_USD),
            rigid_props=preset(
                default=None,
                physx=PhysxRigidBodyCfg(solver_position_iteration_count=PHYSX_SOLVER_POSITION_ITERATIONS),
            ),
            activate_contact_sensors=True,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            # ロボットの爪の下（前方約26cm）に配置
            pos=(0.36, 0.0, mdp.BALL_REST_HEIGHT),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    # プリミティブ生成の上開き箱（11cm x 11cm x 高さ5cm、ロボットの背面側に配置）
    box = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box",
        spawn=OpenBoxSpawnCfg(size=0.11, height=0.05, wall_thickness=0.006),
        init_state=RigidObjectCfg.InitialStateCfg(
            # ロボット(0.10, 0.0)から見て背面側（斜め後ろ: shoulder_panの可動域内で無理なく届く位置）
            pos=(-0.04, 0.12, 0.035),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    desk = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Desk",
        spawn=sim_utils.UsdFileCfg(usd_path=str(DESK_USD), func=_spawn_usd_with_contact_material),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.22, 0.0, 0.032), rot=(0.0, 0.0, 0.7071068, 0.7071068)),
    )

    support = preset(
        default=None,
        sim2real=RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Support",
            spawn=SupportSpawnCfg(
                size=(0.39, 0.39, 0.005),
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                mass_props=sim_utils.MassCfg(mass=1.0),
                physics_material=WORKSHOP_CONTACT_MATERIAL,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.07, 0.07, 0.07)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.22, 0.02, 0.0325)),
        ),
    )

    # Unfiltered: OV PhysX cannot build a filtered contact view for the gripper link in a cloned scene.
    fixed_jaw_contact = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/gripper", history_length=4)
    moving_jaw_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/moving_jaw_so101_v1",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Ball"],
        history_length=4,
    )
    fixed_jaw_support = preset(
        default=None,
        sim2real=_jaw_support_sensor("gripper"),
    )
    moving_jaw_support = preset(
        default=None,
        sim2real=_jaw_support_sensor("moving_jaw_so101_v1"),
    )

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=300.0, color=(0.9, 0.9, 0.9)),
    )
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(intensity=500.0, color=(0.9, 0.9, 0.9), angle=10.0),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.32505758, 0.32505758, 0.0, 0.88807383)),
    )


@configclass
class ActionsCfg:
    """Joint targets relative to the measured position, matching the real SO-101 position interface."""

    arm_action = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINTS,
        preserve_order=True,
        # Small offsets stall the real shoulder and elbow, so sim2real gives them larger steps.
        scale=preset(
            default=0.033,
            sim2real={
                "shoulder_pan": 0.033,
                "shoulder_lift": 0.12,
                "elbow_flex": 0.07,
                "wrist_flex": 0.033,
                "wrist_roll": 0.033,
            },
        ),
        use_zero_offset=True,
    )
    gripper_action = RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["gripper"],
        # Avoid opening a grasp rapidly from a small policy bias.
        scale=0.02,
        use_zero_offset=True,
    )


@configclass
class PolicyStateGroupCfg(ObsGroup):
    """Fully observed state actor inputs."""

    joint_pos = ObsTerm(func=mdp.joint_pos, params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS)})
    joint_vel = ObsTerm(func=mdp.joint_vel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS)})
    joint_target = ObsTerm(func=mdp.joint_target, params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS)})
    previous_action = ObsTerm(func=mdp.last_action)
    end_effector = ObsTerm(
        func=mdp.body_state_b,
        params={
            "body_asset_cfg": SceneEntityCfg("robot", body_names="gripper"),
            "base_asset_cfg": SceneEntityCfg("robot"),
        },
    )
    ball = ObsTerm(
        func=mdp.body_state_b,
        params={"body_asset_cfg": SceneEntityCfg("ball"), "base_asset_cfg": SceneEntityCfg("robot")},
    )
    # 箱（カゴ）の位置観測を追加
    box = ObsTerm(
        func=mdp.body_state_b,
        params={"body_asset_cfg": SceneEntityCfg("box"), "base_asset_cfg": SceneEntityCfg("robot")},
    )
    # 進捗フラグ（把持・リフト・箱投入のマルコフ性維持用）
    progress = ObsTerm(func=mdp.progress_flags)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class CriticStateGroupCfg(PolicyStateGroupCfg):
    """Privileged training critic inputs."""

    contact = ObsTerm(func=mdp.jaw_contact_flags)


@configclass
class ObservationsCfg:
    policy: PolicyStateGroupCfg = PolicyStateGroupCfg()
    critic: CriticStateGroupCfg = CriticStateGroupCfg()


@configclass
class EventsCfg:
    """ボールの物理パラメータ設定およびエピソードリセット設定。"""

    ball_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "static_friction_range": preset(default=VIAL_FRICTION_RANGE, physx=PHYSX_VIAL_FRICTION_RANGE),
            "dynamic_friction_range": preset(default=VIAL_FRICTION_RANGE, physx=PHYSX_VIAL_FRICTION_RANGE),
            "restitution_range": (0.0, 0.02),
            "num_buckets": 32,
        },
    )
    ball_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "mass_distribution_params": (0.015, 0.025),
            "operation": "abs",
        },
    )
    # 机の上にボールをランダム配置してロボットを初期姿勢にするリセット
    # ※下の "grasp_prob": 0.3 をコメントアウト（または 0.0）にすれば、把持スタートをOFF（100%机スタート）にできます
    reset_ball = EventTerm(
        func=mdp.reset_ball_step1,
        mode="reset",
        params={
            "ball_pos_center": (0.36, 0.0),
            "ball_pos_noise": (0.03, 0.04),
            "grasp_prob": 0.3,  # ← ここをコメントアウトすれば把持スタートOFF
        },
    )


def _material_event(asset_cfg: SceneEntityCfg, friction_range: tuple[float, float]) -> EventTerm:
    return EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
            "asset_cfg": asset_cfg,
            "static_friction_range": friction_range,
            "dynamic_friction_range": friction_range,
            "restitution_range": (0.0, 0.02),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )


@configclass
class Sim2RealEventsCfg(EventsCfg):
    """Per-episode contact, motor, calibration and placement variation for robot transfer."""

    ball_material = _material_event(SceneEntityCfg("ball"), (0.2, 1.3))
    jaw_material = _material_event(SceneEntityCfg("robot", body_names=["gripper", "moving_jaw_so101_v1"]), (0.2, 1.3))
    support_material = _material_event(SceneEntityCfg("support"), (0.3, 1.3))
    gripper_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["gripper"]),
            "stiffness_distribution_params": (0.6, 1.7),
            "damping_distribution_params": (0.7, 1.5),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    arm_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.7, 1.5),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    joint_parameters = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS),
            "friction_distribution_params": (0.6, 2.5),
            "armature_distribution_params": (0.7, 1.5),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    joint_viscous_friction = EventTerm(
        func=mdp.randomize_joint_viscous_friction,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=JOINTS), "scale_range": (0.6, 3.5)},
    )
    encoder_bias = EventTerm(func=mdp.randomize_encoder_bias, mode="reset", params={"bound": 0.01})

    def __post_init__(self):
        self.ball_mass.mode = "reset"
        self.ball_mass.params["mass_distribution_params"] = (0.012, 0.030)


@configclass
class RewardsCfg:
    """ボール把持＆箱投入（Pick & Place）報酬設計"""

    # ボール中心への接近報酬（球体中心を狙うためオフセットは0）
    approach_progress = RewTerm(
        func=mdp.approach_progress, weight=1.0, params={"height_offset": 0.0}
    )
    # 持ち上げシェーピング報酬: ボールを浮かせている間、箱の壁を越える目標高さに近づくほど加点
    lift_progress = RewTerm(func=mdp.lift_progress, weight=0.3, params={"pinch_bonus": 0.1})
    # 箱誘導報酬: 安全高さまで持ち上げた後、壁を越えて箱の中へ運ぶほど加点（ホバリング稼ぎを防ぐため重み0.15）
    place_goal = RewTerm(func=mdp.place_goal, weight=0.15, params={"std": 0.12})
    # マイルストーン報酬: 把持(+10.0), 安全高さリフト(+20.0), 箱投入＆リリース(+40.0) の1回限りボーナス
    milestones = RewTerm(
        func=mdp.milestones, weight=1.0, params={"grasp_bonus": 10.0, "lift_bonus": 20.0, "place_bonus": 40.0}
    )
    # 完了ボーナス: 箱の中で安定静止できた時の成功報酬
    success = RewTerm(func=mdp.success, weight=200.0)
    ball_lost = RewTerm(
        func=mdp.is_terminated_term, weight=-50.0, params={"term_keys": ["ball_out_of_bounds", "ball_dropped"]}
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.002)
    joint_velocity = RewTerm(func=mdp.joint_vel_l2, weight=-0.0002)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    # 机押し付けペナルティ: defaultではSupportがないためNone、sim2realプリセット時のみ有効
    jaw_support_force = preset(
        default=None,
        sim2real=RewTerm(func=mdp.jaw_support_force, weight=-8.0, params={"free_force": 0.25, "force_span": 2.0}),
    )


@configclass
class TerminationsCfg:
    success = DoneTerm(func=mdp.ball_place_success)
    ball_dropped = DoneTerm(func=mdp.ball_dropped)
    # ロボットのベース座標系はZ軸周りに+90度回転しているため、机の範囲外を判定
    ball_out_of_bounds = DoneTerm(
        func=mdp.out_of_bound,
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "in_bound_range": {"x": (-0.55, 0.45), "y": (-0.62, 0.62), "z": (-0.01, 1.0)},
        },
    )
    unstable_robot = DoneTerm(func=mdp.abnormal_robot_state)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class PhysicsCfg(PresetCfg):
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=300,
            nconmax=200,
            cone="pyramidal",
            impratio=10.0,
            update_data_interval=2,
            iterations=50,
            ls_iterations=50,
            use_mujoco_contacts=False,
            ccd_iterations=35,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(),
        num_substeps=2,
        debug_mode=False,
    )
    physx = PhysxAutoCfg(
        isaacsim_physx=PhysxCfg(bounce_threshold_velocity=0.01),
        ovphysx=OvPhysxCfg(enable_external_forces_every_iteration=False),
    )
    default = newton_mjwarp


@configclass
class SO101VialEnvCfg(ManagerBasedRLEnvCfg):
    """State task trained from physics-validated reset poses."""

    scene: SO101SceneCfg = SO101SceneCfg(num_envs=4096, env_spacing=0.9, replicate_physics=True)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = preset(default=EventsCfg(), sim2real=Sim2RealEventsCfg())
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        # The measured shoulder response is slower, so sim2real attempts get more time.
        self.episode_length_s = preset(default=20.0, sim2real=30.0)
        self.is_finite_horizon = False
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysicsCfg()
        self.sim.default_visualizer_cfg = VisualizerCfg(eye=(0.64, 0.0, 0.36), lookat=(0.19, 0.02, 0.075))

    def play_mode(self):
        """Play and evaluate complete episodes from the home pose, in dataset order.

        The nominal task pins the vial to its workshop material and mass; sim2real keeps its physics randomization.
        """
        from isaaclab_tutorial.utils import evaluation

        requested_num_envs = self.scene.num_envs
        super().play_mode()
        if evaluation.EXACT_EVALUATION_ACTIVE:
            self.scene.num_envs = min(requested_num_envs, evaluation.EVALUATION_EPISODES)
        else:
            self.scene.num_envs = min(self.scene.num_envs, 16)
        # 評価・再生モード（play）では全ての環境を初期姿勢（机上のボール）から開始する
        self.events.reset_ball.params["grasp_prob"] = 0.0
        if not isinstance(self.events, Sim2RealEventsCfg):
            self.events.ball_material.params.update(
                static_friction_range=(0.7, 0.7), dynamic_friction_range=(0.7, 0.7), restitution_range=(0.0, 0.0)
            )
            self.events.ball_mass.params["mass_distribution_params"] = (0.020, 0.020)
