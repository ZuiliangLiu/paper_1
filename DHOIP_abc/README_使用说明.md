# 混卤候选材料预测与 SHAP 绘图

本包直接加载已有的三个模型，没有重新训练或修改模型。

## 在 PyCharm 中运行

1. 完整解压压缩包，用 PyCharm 打开 `DHOIP_abc` 文件夹。
2. 建议使用 Python 3.12；在该项目终端中安装依赖：

```bash
python -m pip install -r requirements.txt
```

3. 直接运行 `generate_shap_figures.py`，计算 SHAP、保存图片和 CSV，并弹出两版 SHAP 合并图。
4. 运行 `generate_dhoip_abc.py`，调用三个模型完成筛选、预测，并生成前一张参考图的 a、b、c 散点图。

如果只需要保存、不弹出窗口：

```bash
python generate_shap_figures.py --no-show
python generate_dhoip_abc.py --no-show
```

脚本以自身所在文件夹定位数据和模型，不依赖 PyCharm 的工作目录。所有结果保存在 `results`。CSV 均为 UTF-8 BOM 编码，可用 Excel 或 pandas 打开；空白表示该候选材料没有进入对应计算步骤，不表示零带隙。

## SHAP 图片

- `SHAP_used_features.png`：采用当前模型实际选用特征的四张图；另附 PDF 和 a、b、c、d 四张单独 PNG。
- `SHAP_reference.png`：与论文截图采用相同特征的四张图。由于当前模型没有选用原图 b、d 的特征，这两张图的直接 SHAP 贡献为零，并在图中标明。

| 面板 | 论文原特征版本 | 当前模型实际选用特征版本 |
| --- | --- | --- |
| a，结构分类 | r_B + r_B′ | r_B + r_B′ |
| b，结构分类 | r_A − r_B，未选用，SHAP=0 | r_A / r_B |
| c，带隙回归 | χ_B + χ_B′ | χ_B + χ_B′ |
| d，带隙回归 | χ_B′ − χ_X2，未选用，SHAP=0 | χ_B′ − χ_X1 |

灰色实心点代表候选材料；橙色空心点代表已知训练集与测试集材料。结构模型解释 1398 条已知记录和全部 6990 条候选记录；带隙回归模型解释 392 条真实钙钛矿且 DFT PBE 带隙大于 0.2 eV 的已知记录，以及通过两级分类的 2015 条候选记录。

采用 SHAP 0.50.0 的 `TreeExplainer`，`feature_perturbation="tree_path_dependent"`，`model_output="raw"`。预处理与保存的模型完全一致；类别列先独热编码，再将其各编码列 SHAP 值相加，得到该原始类别输入的合并贡献。图中的数值特征各对应一个编码列。

结构分类的 SHAP 单位是对钙钛矿正类的 log-odds 贡献，不是概率或百分数。带隙回归的 SHAP 单位是 eV。两者分别满足：

```text
结构模型基准值 + 所有特征SHAP之和 = decision_function 输出
回归模型基准值 + 所有特征SHAP之和 = 预测PBE带隙
```

已对全部被解释记录验证上述加和关系，最大绝对误差小于 1e-14。结构模型基准值约为 0.913766 log-odds；回归模型基准值约为 2.186721 eV。详情见 `SHAP_metadata.json`。

黑线仅为已知训练与测试记录的普通最小二乘直线拟合，展示总体关系，不是模型本身，也不代表因果规律。SHAP 解释的是模型如何使用输入特征；相关的和、差、比值描述符可能分担贡献。

当前数据集使用正的 Pauling 电负性，图中保留实际数值，没有为了与论文的负坐标或趋势相似而改符号、改预测值或制造点。

## 前一张参考图的 a、b、c

- `figure_a.png`：全部 6990 条候选记录的结构预测类别；横轴 r_B+r_B′，纵轴 r_A/r_B。
- `figure_b.png`：392 条已知记录的真实 DFT PBE 带隙，与 2015 条候选记录的模型预测带隙对比；横轴 χ_B+χ_B′。这是分布比较，不是无标签候选集的准确率检验。
- `figure_c.png`：同一批 2015 条候选记录，用 X 位平均半径着色。平均半径为 (r_X1+r_X2+r_X3)/3，因为每个槽位代表两个卤素原子。
- `figure_abc.png` 和 PDF：三图合并版。

实际筛选结果：6990 条候选记录中，4560 条被预测为钙钛矿，2430 条被预测为非钙钛矿；在前者中，2015 条通过 Eg>0.2 eV 分类，2545 条被分类为 Eg≤0.2 eV。只对通过两级分类的记录进行带隙回归，预测范围约为 0.326984–5.736660 eV。本次没有发现“分类通过但回归值≤0.2 eV”的记录。

6990 和 2015 的计数单位是卤素成对位点构型，并非互不重复的化学组成；2015 条筛选记录对应 990 种化学组成。同一化学式的不同卤素排列保留为不同记录。

## CSV 数据

- `candidate_predictions.csv`：全部候选材料的分阶段预测、类别概率及绘图变量。
- `screened_bandgap_predictions.csv`：通过两级分类并完成带隙回归的 2015 条记录。
- `figure_b_known_DFT.csv`：用于 b 图比较的 392 条已知记录及其原始 train/test 划分。
- `screening_summary.csv`：各阶段数量。
- `structure_SHAP.csv`：结构模型的特征值、合并 SHAP 值、基准值和被解释的原始模型输出。
- `bandgap_regression_SHAP.csv`：带隙回归模型的对应数据。
- `SHAP_feature_status.csv`：两版 SHAP 图的特征对应关系及是否被当前模型选用。

`value: ` 前缀列表示特征原始取值，`SHAP: ` 前缀列表示其模型贡献。`dataset` 区分已知材料与待预测候选，`partition` 保存已知材料原有的 train/test 划分。候选记录没有真实结构和带隙标签，不能据此计算候选集 AUC、RMSE 等误差指标。

## 当前模型的适用范围

当前三个模型的最终选用特征都不含 X2、X3 相关输入。因此一些仅改变 X2、X3 的候选构型会得到相同的预测和 SHAP 值，灰点与橙色空心点可能大量重合。未添加随机抖动或噪声来制造差异。

混卤候选结果用于探索性初筛，需要真实混卤材料的 DFT 结果进行独立验证。结构分类结果不等同于已经确认热力学、动力学或热稳定性。没有根据已有预测值重新训练模型。

## 来源

数据与模型：先前整理的 `predict(1).csv` 与 `DHOIP_three_trained_models.zip`。三份 joblib、已知训练数据和划分文件在此包内保留原字节。

图形参考：Nanoscale 2021，DOI: 10.1039/d1nr01117k，Fig. 3 和 Fig. 4。

SHAP 方法文档：https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html

输入哈希与运行结果见 `run_metadata.json`；SHAP 参数、加和误差和直线拟合参数见 `SHAP_metadata.json`。包内文件校验值见 `SHA256.json`。
