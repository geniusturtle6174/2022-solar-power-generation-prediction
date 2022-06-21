# 2022-solar-power-generation-prediction

My implementation and sharing of this contest: https://aidea-web.tw/topic/09679060-518a-4e6f-94db-53c7d8de8138. I got rank 5 (out of 179 teams) in the Public Leaderboard. The Private Leaderboard has not been announced yet.

## Run My Implementation

### Required libs

`numpy`, `sklearn`, and `xgboost`. Versions of them are not restricted as long as they're new enough. `matplotlib` is also used, but it's for data observation only, and is not required for model training and inference. Besides, if you want to try my deep learning methods (which may not be effective in this contest) in the `legacy` directory, then `torch` and `yaml` are needed.

### Training
```bash
python3 train_xgb.py save_model_dir --n_fold 7
```
* `save_model_dir`: where you want to save the trained model.
* `--n_fold`: number of folds for cross validation. Default to 7.
* Input csv files are assumed to be in the `data` dir.

### Inference
```bash
python3 test_xgb.py model_dir --output_file_name submission.csv --n_fold_train 7
```
* `model_dir`: directory of the trained model.
* `--output_file_name`: output file name for submission.
* `--n_fold_train`: number of folds used while training.

## 作法分享

以下將介紹本競賽的問題概述，以及所使用的執行環境、特徵截取、模型設計與訓練，以及預測方式。

### 問題概述

* 給定資料：約一年期間內，數個太陽能發電場域的經緯度、天氣觀測資料、發電模組規格，以及發電量等。
  * 每個發電廠域的資料起始時間不盡相同，結束時間皆為 2021/10/28。
  * 發電場域位於彰化及桃園的數個不同鄉鎮市區。
* 預測目標：2021/10/29 起至 2022 年二月下旬止，每場域的每日發電量。
  * 不同場域的截止日期不同，位於彰化者皆為 2022/2/16，位於桃園者皆為 2022/2/17
* 評估標準：RMSE。

### 執行環境

硬體方面為 ASUS P2440 UF 筆電，含 i7-8550U CPU 及 MX130 顯示卡，主記憶體擴充至 20 GB。程式語言為 Python 3，函式庫則如本說明前半部所示，皆未特別指定版本。

### 特徵擷取

於本比賽中，除了大會給定的資料外，亦有使用外部資料。對於兩種資料的特徵擷取，分別介紹如下。

#### 給定資料

從大會給定的資料中，抽出特徵共 59 維，細節如下：
* 可能缺值類共 6 維: Temp_m, Irradiance, Temp 各 2 維，一維代表是否缺值，另一維為實際值
* 模組相關共 10 維: 模組 one-hot 共 4 維，以及模組規格（峰值輸出等）共 6 維
* 經緯度相關 11 維: 發電廠域經緯度的 one-hot
* 時間相關 9 維: 月份、月份以三月為第一月、月份以六月為第一月、月份以九月為第一月、是否為春季（三到五月）、是否為夏季（是否為六到八月）、是否為秋季（九到十一月）、是否為冬季（十二到二月）、日期
* 其餘原始特徵及角度特徵工程 10 維: 裝置容量、發電廠域經度、發電廠域緯度、角度、角度正弦值、角度餘弦值、角度雙取正切值、角度雙取正切值取負號再加一、角度正負號（零度設為正號）、日照計之日射量
* 其餘特徵工程 13 維:
  * 1 維: 裝置容量乘以日照計之日射量除以一千
  * 5 維: 日照計之日射量分別乘以一減角度、角度正弦值、角度餘弦值、角度雙曲正切值、角度雙曲正切值取負號再加一
  * 5 維: 裝置容量乘以前述 5 維
  * 2 維: 模溫計之模板溫度除以當日平均氣溫，以及日射量除以日照計之日射量，遇有缺值時皆填零

#### 外部資料

外部資料的來源為氣象局的[觀測資料查詢](https://e-service.cwb.gov.tw/HistoryDataQuery/index.jsp)，測站選擇的方式是離發電廠域最接近的，且有觀測資料的測站（手動準備好測站座標，以 `calc_lat_lon.py` 計算離某發電廠域最近的測站）；資料格式為月報表，即逐日的觀測資料。特徵的列表如下，共 39 維；代號所指的觀測項目和單位等，請自行參考報表：
* 測站類 5 維: 測站 one-hot
* 氣壓類 4 維: StnPres, SeaPres, StnPresMax, StnPresMin
* 溫度類 4 維: Temperature, T Max, T Min, Td dew point
* 濕度類 2 維: RH, RHMin
* 風力類 4 維: WS, WD, WSGust, WDGust
* 降水類 4 維: Precp, PrecpHour, PrecpMax10, PrecpMax60
* 日照及其他類 7 維: SunShine, SunShineRate, GloblRad, VisbMean, EvapA, UVI Max, Cloud Amount
* 特徵工程 9 維: SunShine, SunShineRate, 以及 GloblRad 分別乘以 angle 的正弦值、angle 的雙曲正切值，以及 1 - angle

#### 其他說明

* 特徵的部份設計，如多一維表示缺值，或者除以常數等，是因早先使用深度學習模型時而設計，而模型轉換為 XGBRegressor 仍將其保留，並未拆除。
* 嘗試過其他特徵工程，但在嘗試當下的模型參數設定下，沒有取得比當時最佳成果好的結果。
* 使用過兩個測站，根據測站選擇與特徵使用（單獨使用或以不同方式合併）方式的不同，效果與一個測站的相比，會稍差或相同。未嘗試過使用三個以上的測站資料。
* 使用過討論區參賽者提供的[這個工具](https://github.com/JackyWeng526/Taiwan_Weather_Data)來下載日報表，以加入逐小時的觀測資料為特徵，但沒有取得比較好的效果。

### 模型設計、訓練與觀察

#### 模型設計與訓練

本次比賽使用的模型為 XGBRegressor，訓練方式為 n folds cross validation。模型的細節參數請參考 `train_xgb.py` 的 `param` 變數，未設定之參數係依照預設值，未進行修改。所有模型的預測目標皆是直接輸出每場域每日的發電量，沒有另外作正規化等調整。

切 fold 的方式分為以下兩種，實驗結果為以下兩種方式都使用，來訓練出 2n 個模型，效果會稍微好一些：
1. 每個場域各自依照時間區段來切。例如某場域有十個月的資料要切成 5 個 folds，則前兩個月為 fold 0，接下來的兩個月為 fold 1，依此類推。
2. 先將全部訓練資料依照發電量排序，再根據索引值除以 fold 數的餘數來切。例如要切成 5 個 folds，則排序後的位置為 0, 5, 10, ... (餘數 0)者為 fold 0，位置為 1, 6, 11, ... (餘數 1)者為 fold 1，依此類推。

另外，由於 XGBRegressor 並不會自動回傳達到最佳 validation loss 的模型，因此我先用 `param` 變數當中所示的參數訓練一次，待取得 validatino loss 的曲線後，再根據最佳 loss 的位置來重新設定 `n_estimators` 的值，並重新訓練一次，來當作最佳 validation loss 的模型來使用。

由上述方式產生出的 2n 個模型，都會做為預測使用。另外，在使用 XGBRegressor 之前，我先使用了深度學習，但效果不是很理想。

#### Feature Importance 觀察

不同參數訓練出的模型，以及不同 fold 當中，比較重要的特徵可能不盡相同，但大致可觀察到這幾個現象：
* 裝置容量和日射量穩定佔據前兩名，且裝置容量占據五成以上的重要性，而在早期未引入外部資料時，甚至有觀察到裝置容量占了八成左右的比重。
* 內部及外部有關日射量的特徵工程，有少部分明顯的佔據接下來的幾名，個別的重要性約 1% 至 10%。事實上，在僅使用內部資料時已觀察到相關現象，故外部資料的特徵工程也參考了此現象來設計。
* 發電模組規格、季節，以及部分的角度特徵工程等特徵，其重要性為 0，可能是模型已從其他特徵上學習到相關資訊，例如模組的 one-hot 已經隱含地包含了模組規格，故模型可能就不再需要更細節的規格資訊。

### 預測

預測時會將 2n 個模型的結果，取中位數做為最終輸出。

我亦嘗試過使用平均值，或者只使用部分模型等其他方式來產生最終輸出，例如根據 validation loss 設定權重或者去除表現較差的模型，但是都沒有達到比較好的效果。亦可以經由 `test_xgb.py` 的 `--n_fold_train` 參數，帶入比訓練時的 folds 數目少的數字，來達到只使用前 k 個模型來預測的效果，但並未於實驗中測試過。

### 心得

* 有些調整參數的方向比較晚開始嘗試，但是最後期限已近，所剩的上傳次數不夠我再嘗試，因此模型應該還有微幅調整的空間；雖然光是靠調參數，應該也只會有微幅變化，比較難產生決定性的影響。
* 本次競賽比較特別的是，設立了一個「持續領先獎」，也就是佔據排行榜第一名最長時間的獎項，但比較可惜的是，這個獎項搭配上了 AIdea 平台以最後上傳而非最佳結果為準的設計，可能會讓佔據第一名的參賽者怯於持續調整精進，我認為平台在這方面有改進空間。
