# 四足机器人状态估计中的坐标变换（基于新版 `state_estimator.py`)

本文对照 `locomotion/estimator/state_estimator.py`（无 FSM、无 RL）说明坐标系与变换链，保持与现行实现一致。

## 坐标系
- **World**：全局系，Z 向上。  
- **Body**：机体系，原点在质心，Z 向上。  
- **Yaw Frame**：仅保留 yaw 的中间系，Z 与 World 对齐。  
- **Ground Frame**：对齐地面法向，Z 垂直地面。  

## 主要状态量（同名字段）
- `pos`：质心位置（初始化为 `_bodyHeight`）。
- `ori`：姿态四元数 `[x,y,z,w]`。
- `v_world` / `omega_world`：世界系速度 / 角速度。
- `v_body` / `omega_body`：机体系速度 / 角速度。
- `body_R_world`：World → Body 的旋转矩阵（`quat_to_rot` 结果，直接右乘世界系向量）。
- `rpy`：世界系 RPY。
- `rpy_body`：Ground Frame 下的 RPY。
- `ground_normal_yaw` / `ground_normal_world`：地面法向在 yaw / ground 与 world 中的表示。
- `ground_R_body`：Body → Ground 的旋转。

## 更新流程（`update()`）
1. **世界系速度/角速度**
   ```python
   for i in range(3):
       v_world[i]     = body_states["vel"]["linear"][i]
       omega_world[i] = body_states["vel"]["angular"][i]
   ```
2. **姿态四元数**
   ```python
   ori = [x,y,z,w] from body_states["pose"]["r"]
   ```
3. **四元数 → 旋转矩阵（World→Body）**
   ```python
   body_R_world = quat_to_rot(ori)
   ```
4. **速度转到机体系**
   ```python
   v_body     = body_R_world @ v_world
   omega_body = body_R_world @ omega_world
   ```
5. **世界系 RPY**
   ```python
   rpy = quat_to_rpy(ori)
   ```
6. **构建 Yaw / Ground 旋转**
   ```python
   yaw_R_world   = rpy_to_rot([0,0,rpy[2]])
   ground_R_yaw  = get_rot_from_normals([0,0,1], ground_normal_yaw)
   ground_R_body = body_R_world @ yaw_R_world.T @ ground_R_yaw.T
   ```
7. **Ground 系 RPY**
   ```python
   rpy_body = rot_to_rpy(ground_R_body)
   ```
8. **地面法向**
   - 平地：`ground_normal_yaw = [0,0,1]`
   - 斜坡/不平地：`_compute_ground_normal_and_com_position()` 用接触足最小二乘估计法向；`ground_normal_world = body_R_world.T @ ground_normal_yaw`。

## 接触与高度
- `_contact_phase`：4x1，1=接触，0=摆动。  
- `_init_contact_history`：首次用当前足端，Z 置 `-body_height`。  
- `_update_contact_history`：仅对接触腿写入历史。  
- `_update_com_position_ground_frame`：  
  - 无接触腿：返回默认高度；  
  - 有接触腿：将足端从 Body 旋到 Ground，取接触足 Z 的加权平均赋给 `pos[2]`。  
- `_compute_ground_normal_and_com_position`：在更新高度后，对接触足做最小二乘求法向，法向朝上（Z>0）。  

## 变换链速览
```
World →(quat_to_rot) body_R_world → Body
World →(yaw_R_world) → Yaw
Yaw   →(ground_R_yaw)→ Ground
Body  → Ground: ground_R_body = body_R_world @ yaw_R_world.T @ ground_R_yaw.T
```
速度使用 `body_R_world` 右乘完成 World→Body 变换；姿态在 Ground 下通过 `ground_R_body` 获得 `rpy_body`。

## 使用要点
- 平地可保持 `ground_normal_yaw=[0,0,1]`；斜坡/楼梯需启用接触估计。  
- 传入的速度、角速度需为世界系量，并与仿真重力方向一致（注意仿真重力符号）。  
- 若初始化阶段 `ground_R_body` 为空，系统依赖默认高度与法向，避免控制发散。  

