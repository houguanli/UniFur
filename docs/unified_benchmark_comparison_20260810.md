# 統一評測：Residual-only、Unified 與外部方法

日期：2026-08-10

## 結論

目前唯一可以完整、直接排名的動態對照，是相同 DFA Panda walk 資料、相同模板與骨架、相同 neutral 20k Gaussian 初始化、相同 20k steps，以及相同 8 個 held-out cameras 上的 Residual-only 與 Unified-soft。

這組實驗不支持「目前 Unified 已經勝過 Residual-only」的結論。Unified 在單目輸入下改善輪廓與感知品質，但沒有改善 PSNR；在 4/8 視角下幾乎持平或略退，訓練時間為 Residual-only 的 1.75–1.84 倍。最終路由仍有 96.36–98.84% 分配到 residual，結構化 shell/strand experts 尚未形成有效、穩定的分工。

## 凍結的動態評測標準

- Protocol ID：`DFA-Panda-Walk-32f-v1`。
- 資料：Artemis Dynamic Furry Animals，Panda walk，32 個同步時間幀，960×540 RGBA。
- D-mono：camera 1，共 32 張訓練影像。
- D-mv4：cameras 1/11/24/34，共 128 張訓練影像。
- D-mv8：cameras 1/6/11/16/19/24/29/34，共 256 張訓練影像。
- 共同測試：cameras 0/5/10/15/20/25/30/35 × 32 幀，共 256 張 held-out-view images。
- 共同先驗：DFA furless body、93-bone skeleton、exact matrix LBS；20k neutral-gray surface Gaussians，不使用 held-out views 的 appearance。
- 共同計算：20k source roots、20k optimization steps、同一 rasterizer 與損失定義。
- 指標：FG PSNR、masked PSNR、full-image PSNR/SSIM/LPIPS、mask IoU、background opacity、訓練時間與峰值 CUDA reserved memory。

## 同口徑動態主表

| Input | Method | FG PSNR ↑ | Masked PSNR ↑ | Full PSNR ↑ | SSIM ↑ | LPIPS ↓ | IoU ↑ | BG alpha ↓ | Train s ↓ | Peak GB ↓ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mono | Residual-only | **11.6367** | **19.2927** | **17.1401** | 0.76095 | 0.26323 | 0.76150 | 0.06811 | **250.2** | **8.66** |
| mono | Unified-soft | 11.5823 | 19.2383 | 17.1352 | **0.76626** | **0.25840** | **0.76818** | **0.06477** | 437.9 | 8.68 |
| mv4 | Residual-only | 19.8885 | 27.5445 | **24.3269** | 0.90431 | **0.12558** | **0.95203** | 0.01017 | **247.2** | **8.72** |
| mv4 | Unified-soft | **19.9051** | **27.5611** | 24.2808 | **0.90479** | 0.12562 | 0.95200 | **0.01015** | 443.1 | 8.73 |
| mv8 | Residual-only | **23.7418** | **31.3978** | **28.0688** | 0.93179 | 0.11856 | **0.97357** | **0.00594** | **241.6** | **8.80** |
| mv8 | Unified-soft | 23.6349 | 31.2909 | 27.9816 | **0.93195** | **0.11795** | 0.97336 | 0.00594 | 443.4 | 8.81 |

粗體只表示同一 input setting 內的較佳值。數值差很小時不應解讀為顯著勝出；下一輪需要至少 3 個 seeds 或更多 sequences 報告平均與標準差。

## 成對差值：Unified-soft − Residual-only

| Input | ΔFG PSNR | ΔFull PSNR | ΔSSIM | ΔLPIPS | ΔIoU | ΔBG alpha | Time × | Shell / Strand / Residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mono | -0.0544 | -0.0048 | +0.00531 | -0.00483 | +0.00668 | -0.00335 | 1.75× | 1.136% / 0.025% / 98.838% |
| mv4 | +0.0167 | -0.0461 | +0.00048 | +0.00004 | -0.00003 | -0.00002 | 1.79× | 2.172% / 1.470% / 96.358% |
| mv8 | -0.1069 | -0.0872 | +0.00015 | -0.00061 | -0.00021 | +0.00000 | 1.84× | 1.808% / 0.724% / 97.469% |

單目下的 LPIPS、IoU 和背景 alpha 改善是真實訊號，但尚不足以抵消 PSNR 不升、compute 增加和 route collapse。多視角輸入本身帶來的收益遠大於 representation 的收益：Residual-only 從 mono 到 mv4 的 FG PSNR 增加 8.25 dB，從 mv4 到 mv8 再增加 3.85 dB。

### 256-image paired bootstrap（20,000 resamples）

| Input | ΔFG PSNR 95% CI | ΔFull PSNR 95% CI | ΔSSIM 95% CI | ΔLPIPS 95% CI | ΔIoU 95% CI |
|---|---:|---:|---:|---:|---:|
| mono | [-0.0730, -0.0361] | [-0.0203, +0.0107] | [+0.00491, +0.00571] | [-0.00556, -0.00411] | [+0.00591, +0.00748] |
| mv4 | [-0.0003, +0.0341] | [-0.0706, -0.0215] | [+0.00037, +0.00058] | [-0.00023, +0.00031] | [-0.00023, +0.00017] |
| mv8 | [-0.1310, -0.0838] | [-0.1066, -0.0682] | [+0.00004, +0.00027] | [-0.00093, -0.00029] | [-0.00033, -0.00009] |

CI 只量化這一條 sequence 的逐幀穩定性，不替代跨 sequence / cross-seed variance。它支持三個判斷：mono 的 perceptual/boundary gain 與 FG PSNR loss 同時成立；mv4 的 FG PSNR 是平手；mv8 的 PSNR 退化成立。

## 靜態 Panda：同一 released body prior 的內部對照

這一表使用 NeuralFur-processed Artemis Panda 的 28 fit / 8 test views、512×288、20k roots、20k steps。兩個內部方法完全同口徑，但 released body Gaussian 的 `eval=False` 表明它曾看過全部 36 views，因此只能稱為 `S-mv-official-prior`，不能稱為嚴格 train-view-only reconstruction。

| Method | FG PSNR ↑ | Masked PSNR ↑ | Full PSNR ↑ | SSIM ↑ | LPIPS ↓ | IoU ↑ | BG alpha ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Residual-only | 24.5151 | 32.2067 | 30.3697 | 0.94476 | 0.09582 | **0.99038** | 0.00255 |
| Unified-soft | **24.7883** | **32.4799** | **30.4969** | **0.94662** | **0.09137** | 0.99005 | **0.00253** |

Unified 在這個較容易且 prior 較強的靜態 case 有小幅增益，但 soft routes 為 0.153% shell、4.429% strand、95.418% residual；仍不是完整的三 expert 自適應成功證據。

## 外部方法：完成狀態與可比性

| Method | Local state | Current number | 是否可進主排名 | 原因 |
|---|---|---:|---|---|
| NeuralFur official 15k | step 1 OOM on RTX 4090 24 GB | OOM | 否 | 官方容量未完成，不能以失敗值排名 |
| NeuralFur scaled 4k | 20k complete，official 28/8 split，1920×1080 | masked test PSNR 12.8161 | 僅工程錨點 | strand 數由 15k 降至 4k；只有官方 masked PSNR/L1/CE/orientation 指標，與內部完整指標不齊 |
| Vidu4D | DFA mono adapter complete | pending | 否 | DFA training 與共同 held-out-camera evaluator 尚未完成；舊 Cat 只保留為環境診斷 |
| GART | environment ready | blocked | 否 | 官方 dog fitting 需要有授權限制的 D-SMAL/BITE assets |
| 4D-Animal | code pulled at `2b8a959` | pending | 否 | released runner/asset 是 CoP3D-specific；DFA adapter 尚未完成，且輸出是 body geometry/motion，不是 fiber renderer |
| AnimalGS | paper available | unavailable | 否 | 目前沒有可重現的官方 code/results |
| HairGS | wCurly local run complete | fitted-view PSNR 30.4869；mask IoU 0.9112；geometry F1 0.5035 @ 4 mm/40° | 否，獨立 hair table | 不同資料、不同任務與幾何閾值，不能和 Panda/DFA PSNR 混排 |

NeuralFur 的 12.8161 是官方程式以 `image * gt_mask` 在 8 個 test cameras 上平均的 PSNR，定義接近本報告的 masked PSNR，但解析度、strand budget 與先驗類別不同，故只作工程診斷。官方論文主要以 fur length、curvature、orientation 和 silhouette Chamfer 等無監督幾何統計比較，不提供一套可直接填入本動態主表的 full PSNR/SSIM/LPIPS。

## 現階段判斷

1. **主線應先押多視角。** mv4 已使 Residual-only 達到 19.89 dB FG PSNR，mv8 達 23.74 dB；這比當前 Unified 的 representation gain 大兩個數量級。這條線最容易先形成可發表的穩定結果。
2. **Residual-only 是目前真正的強 baseline。** Unified 還沒有證明 compute-quality Pareto 改善，不能只靠靜態 Panda 的 +0.273 dB 宣稱成功。
3. **單目可以保留為高價值副線。** Unified 在單目輪廓和 LPIPS 上的改善說明 shell-like boundary regularization 有用，但 appearance/novel-view information 不足，不能僅靠 routing 解決欠約束問題。
4. **下一個最高優先修改是反 route-collapse 的 marginal-contribution training。** 對每個 expert 直接估計 held-out contribution，給 shell/strand 加 orientation、boundary-band、thinness/curvature 的專屬 supervision；同時加入 expert budget，使 Unified 的額外 compute 必須換來可測增益，而不是以 97% residual 產生近似相同輸出。
5. **外部方法完成前不補假數字。** 下一批應優先完成 Vidu4D-DFA-mono 的共同 256-image evaluator；NeuralFur 需要能容納官方 15k 的更大顯存或 memory-reduced faithful implementation，才有正式排名資格。

## 可重現產物

- 動態 protocol：`F:/fur_hair_unified_data/benchmarks/dfa_panda_walk_dual/protocol.json`
- 動態 leaderboard：`F:/fur_hair_unified_data/benchmarks/dfa_panda_walk_dual_results/dual_input_leaderboard.md`
- Machine-readable JSON/CSV：同目錄下的 `dual_input_leaderboard.json` 與 `dual_input_leaderboard.csv`
- 每組 256 張逐幀指標與預覽：`*_eval_novel_v8/evaluation.json`、`frames/` 與 `evaluation_contact_sheet.png`
- NeuralFur 4k checkpoint：`F:/fur_hair_unified_data/benchmarks/neuralfur_panda_shared/neuralfur_4k_full20k_lrbody_r512/checkpoints/20000.pth`

## 方法來源

- NeuralFur：https://arxiv.org/abs/2601.12481
- HairGS：https://yimin-pan.github.io/hair-gs/
- Vidu4D：https://vidu4d-dgs.github.io/
- GART：https://github.com/JiahuiLei/GART
- 4D-Animal：https://openaccess.thecvf.com/content/WACV2026/html/Zhong_4D-Animal_Freely_Reconstructing_Animatable_3D_Animals_from_Videos_WACV_2026_paper.html
