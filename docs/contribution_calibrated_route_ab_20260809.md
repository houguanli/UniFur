# Unified Fiber contribution-calibrated routing：Cat A/B 驗證（2026-08-09）

## 結論

這一輪驗證支持保留「soft mixture 作為最終表示、shell/strand/residual 作為可學習專家」的方向。新增的局部一致性、route dropout 與 held-out contribution calibration，能在幾乎維持重建品質的情況下，顯著改善路由的空間連續性及 probability/contribution 對齊。

它還不能被解讀為每個 Gaussian 的 epistemic confidence，也不是完整 20k 點品質結論；目前證明的是：在固定 Cat 資料、固定訓練/校準/測試切分及完整 1200-step 優化下，路由機制本身比舊版穩定且更可解釋。

## 實作變更

- 最終部署維持 soft routing，不再以 hard argmax 作為主要輸出。
- 在 rest-surface 建立 8-NN 圖，對鄰域 route probability 加入平滑正則。
- 訓練時隨機移除一個 route 並重新歸一化其餘 route；對容易吸收所有殘差的 residual route 提高 dropout 機率。
- 從 fit views 中預留 4 個 calibration views，不參與 photometric fitting。
- 週期性在 4 個 calibration views 上做 leave-one-route-out 渲染，以 loss increase 建立 route contribution target。
- 將 contribution target 與穩定先驗混合，再以 EMA 更新，避免單幀小 loss 造成 residual collapse。
- 加入 deterministic seed、設定合法性檢查、報告欄位及路由訓練單元測試。

## 公平 A/B 協議

| 項目 | 設定 |
|---|---|
| 資料 | Cat 序列，同一份已整理資料 |
| source Gaussians | 1,000 |
| renderer | HairGS |
| resolution | 256 × 144 |
| optimization | 1,200 steps，兩組完整跑完 |
| fit frames | 0–27 |
| calibration frames | 28–31（baseline 同樣保留但不使用） |
| held-out test frames | 32–39 |
| baseline | 原始 route prior／entropy／hardening；無 KNN、dropout、risk calibration |
| candidate | soft routing + 8-NN + biased dropout + 4-view contribution calibration |

## 結果

| 指標 | Baseline-28 | Calibrated soft | 變化 |
|---|---:|---:|---:|
| held-out foreground PSNR ↑ | 14.6130 | 14.5230 | -0.0899 dB |
| held-out L1 ↓ | 0.14733 | 0.14908 | +0.00176 |
| held-out IoU ↑ | 0.40003 | 0.39072 | -0.00932 |
| held-out F1 ↑ | 0.57139 | 0.56186 | -0.00953 |
| probability–contribution TV ↓ | 0.03441 | 0.02567 | **-25.4%** |
| 8-NN hard-route agreement ↑ | 0.50938 | 0.68363 | **+0.17425** |
| 8-NN probability L1 ↓ | 0.27689 | 0.19607 | **-29.2%** |
| normalized route entropy | 0.52831 | 0.59866 | +0.07035 |

Candidate 的 held-out route mass 為 shell/strand/residual = 0.2977/0.1243/0.5780；同一批 test views 的 leave-one-out impact 為 0.3105/0.1371/0.5524。三類差距分別約 -0.0128、-0.0129、+0.0257，比較接近實際渲染貢獻，且沒有再次坍縮到 residual-only。

Candidate 的 hard-vs-soft PSNR gap 從 0.1907 增為 0.4539 dB。這不是主要部署品質倒退，而是明確證據：新模型學到的是混合表示，不能在測試時無代價地硬化為單一路由。因此模型輸出與評估都應以 soft mixture 為主，hard route 僅作診斷和視覺化。

## 失敗實驗與修正

最初版本只用單一 calibration frame，直接用極小的 leave-one-out loss 作 target；結果 route mass 坍縮為 shell/strand/residual = 0.1308/0.0923/0.7770，target 更達到 0.0134/0.0039/0.9827。這證明單幀 risk signal 對遮擋和局部誤差過度敏感。

修正後同時聚合 4 個 calibration frames、加入 prior blend、EMA 與正值 floor，並降低 calibration loss 權重及更新頻率。上述 A/B 是修正後結果。

## 範圍與下一步

本輪受同機另一個長時間 GPU 工作持續佔用約 24 GB VRAM 影響，沒有中止該工作。20k 點 aggregated run 在 warm-up 階段安全停止，保留輸出作診斷；改用 1k 點完成受控 A/B，目的是先驗證機制而非宣稱最終畫質。

下一個有效實驗是在 GPU 空閒後，完全沿用此設定跑 20k 點 Cat，然後在對齊的 hair 資料上跑相同 ablation。若 20k 下 PSNR/IoU 代價仍小於目前量級，同時 KNN 與 contribution 指標保持改善，這套 soft expert routing 才可升格為主線設定。

## 驗證

- 主環境：16 passed，1 skipped。
- HairGS renderer 整合：1 passed。
- A/B 的兩組 1200-step 優化、soft/hard held-out evaluation、route audit 均完整產出。

