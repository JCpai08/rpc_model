# J2000 姿态四元数转换到 ECEF 的开发说明

本文档将 `coord_transform_J2000_ECES.md` 中的数学描述改写为开发者可直接实现的流程说明。

---

## 1. 目标与输入输出

### 目标
将卫星在 **J2000 地心惯性坐标系** 下的姿态四元数，转换为 **ECEF 地固坐标系** 下的姿态四元数。

### 输入
- `q_j2000`：原始姿态四元数（建议约定顺序：`[w, x, y, z]`，需在代码中统一）
- `imaging_time`：成像时刻（UTC）

### 输出
- `q_ecef`：转换后的 ECEF 姿态四元数（归一化）

---

## 2. 整体流程（3 步）

1. 根据成像时刻计算时间参数（`T`、`d`）以及地球定向角（`alpha0`、`delta0`、`W`）
2. 构建 `R_j2000_to_ecef` 旋转矩阵
3. 将姿态四元数转矩阵、做坐标变换、再转回四元数

---

## 3. 时间参数与角度计算

根据原文档公式：

- `alpha0 = 0.00 - 0.641 * T`（单位：度）
- `delta0 = 90.00 - 0.557 * T`（单位：度）
- `W = 190.147 + 360.9856235 * d`（单位：度）

其中：
- `T`：儒略世纪（相对 J2000 历元）
- `d`：儒略日间隔（相对参考历元）

> 实现建议：
> - 在代码里明确 `T` 与 `d` 的基准定义（与现有数据/文档保持一致）。
> - 所有三角函数统一使用弧度，角度先 `deg2rad`。

---

## 4. 构建 J2000 到 ECEF 的旋转矩阵

采用 Z-X-Z 旋转序列：

1. `M1 = Rz(alpha0 + 90°)`
2. `M2 = Rx(90° - delta0)`
3. `M3 = Rz(W)`

合成：

- `R_j2000_to_ecef = M3 @ M2 @ M1`

说明：
- `Rz(theta)` 为绕 Z 轴右手旋转矩阵
- `Rx(theta)` 为绕 X 轴右手旋转矩阵
- `@` 表示矩阵乘法

---

## 5. 姿态变换逻辑

### 5.1 四元数转姿态矩阵

- `R_sat_j2000 = quat_to_matrix(q_j2000)`

### 5.2 坐标系变换

按原文档约定：

- `R_sat_ecef = R_sat_j2000 @ transpose(R_j2000_to_ecef)`

> 这里使用转置是因为旋转方向需要做逆变换（正交矩阵满足 `R^-1 = R^T`）。

### 5.3 转回四元数并归一化

- `q_ecef = matrix_to_quat(R_sat_ecef)`
- `q_ecef = normalize(q_ecef)`

---

## 6. 建议的函数拆分

建议在 `rpc_model/coord_transform.py` 中使用以下职责拆分：

1. `datetime_to_julian_params(imaging_time) -> (T, d)`
2. `build_j2000_to_ecef_matrix(T, d) -> np.ndarray(3, 3)`
3. `quat_to_matrix(q) -> np.ndarray(3, 3)`
4. `matrix_to_quat(R) -> np.ndarray(4,)`
5. `convert_attitude_j2000_to_ecef(q_j2000, imaging_time) -> q_ecef`

这样便于单测覆盖每个环节，快速定位误差来源。

---

## 7. 数值与工程注意事项

1. **四元数顺序必须统一**：`[w, x, y, z]` 或 `[x, y, z, w]` 只能二选一。
2. **输入四元数先归一化**：避免姿态矩阵非正交。
3. **角度周期处理**：`W` 可做 `W % 360`，防止大数累积误差。
4. **矩阵正交性检查**：必要时对 `R_sat_ecef` 做轻量正交化（如 SVD 修正）。
5. **输出符号一致性**：`q` 与 `-q` 表示同一姿态，测试时注意比较方式。

---

## 8. 伪代码

```python
def convert_attitude_j2000_to_ecef(q_j2000, imaging_time):
    q_j2000 = normalize(q_j2000)

    T, d = datetime_to_julian_params(imaging_time)

    alpha0 = 0.00 - 0.641 * T
    delta0 = 90.00 - 0.557 * T
    W = (190.147 + 360.9856235 * d) % 360.0

    M1 = Rz(deg2rad(alpha0 + 90.0))
    M2 = Rx(deg2rad(90.0 - delta0))
    M3 = Rz(deg2rad(W))

    R_j2000_to_ecef = M3 @ M2 @ M1

    R_sat_j2000 = quat_to_matrix(q_j2000)
    R_sat_ecef = R_sat_j2000 @ R_j2000_to_ecef.T

    q_ecef = matrix_to_quat(R_sat_ecef)
    q_ecef = normalize(q_ecef)

    return q_ecef
```

---

## 9. 最小测试建议

建议在 `tests/test_coord_transform.py` 增加以下用例：

1. **单位四元数输入**：验证输出是否有限且归一化。
2. **随机四元数回归**：固定时间点，检查转换结果是否稳定。
3. **矩阵链路一致性**：`quat->matrix->quat` 的往返误差在阈值内。
4. **边界时间**：J2000 历元附近时间点，检查角度计算无异常。

---

## 10. 与原计划文档关系

- 原文档保留理论推导和公式来源。
- 本文档用于工程实现和代码落地，重点是“如何写代码”和“如何测试”。
