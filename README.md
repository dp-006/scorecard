# venv
python -m venv venv
venv\Scripts\activate

# Veri Hazırlama
python .\data_preparation\dataprep.py

---

# Workflow: WOE -> Logistic Regression -> Scorecard

This document explains the end-to-end process of transforming raw data into a scorecard, including the Weight of Evidence (WOE) transformation, Logistic Regression modeling, and the final scorecard generation. Each step is detailed with formulas and examples to ensure clarity.

---

## 1. Weight of Evidence (WOE)

**What is WOE?**  
Weight of Evidence (WOE) is a technique used to encode categorical or continuous variables into a format suitable for Logistic Regression. It measures the predictive power of a feature by comparing the distribution of good and bad outcomes.

**Formula:**

```
WOE = ln(%Good / %Bad)
```

Where:
- %Good: Proportion of good outcomes in the bin.
- %Bad: Proportion of bad outcomes in the bin.

**Example:**

| Bin  | Good | Bad | %Good | %Bad | WOE   |
|------|------|-----|-------|------|-------|
| 1    | 50   | 10  | 0.833 | 0.167| 1.61  |
| 2    | 30   | 20  | 0.600 | 0.400| 0.51  |
| 3    | 20   | 70  | 0.222 | 0.778| -1.25 |

---

## 2. Logistic Regression

**What is Logistic Regression?**  
Logistic Regression is a statistical method used to model the probability of a binary outcome (e.g., good vs. bad). It predicts the log-odds of the outcome as a linear combination of the input features.

**Formula:**

```
z = b0 + b1*x1 + b2*x2 + ... + bn*xn
```

The probability of the positive outcome is:

```
P(Good) = exp(z) / (1 + exp(z))
```

**Example:**

Using the equation:

```
z = 0.0525 - 0.3891*feature_1 + 0.0641*feature_2 - 1.2422*feature_3
```

If feature_1 = 0, feature_2 = 0, feature_3 = 1:

```
z = 0.0525 - 1.2422 = -1.1897
P(Good) = exp(-1.1897) / (1 + exp(-1.1897)) = 0.234
```

---

## 2. Lojistik Regresyon Modelinde logit(p) ve Odds

> **Lojistik Regresyon modelimizde, model çıktısı (p) bir gözlemin "1" yani "Kötü" (Bad) olma olasılığıdır.**
>
> logit(p) ifadesi, bu olasılığın odds'unun (olasılık/karşı olasılık) logaritmasıdır:
>
> ```
> logit(p) = ln(p / (1 - p))
> ```
>
> Burada:
> - p: Bir gözlemin "Kötü" (Bad, y=1) olma olasılığıdır.
> - (1-p): Aynı gözlemin "İyi" (Good, y=0) olma olasılığıdır.
> - p / (1-p): "Kötü" olma odds'u, yani bir gözlemin kötü olma ihtimalinin iyi olma ihtimaline oranı.
>
> Modelin tahmin ettiği değerler, bu odds'un logaritmasıdır. Yani model, her gözlem için "Kötü" olma odds'unu tahmin eder.

---

## 3. Scorecard Generation

**What is a Scorecard?**  
A scorecard is a representation of the Logistic Regression model where the log-odds are converted into a user-friendly score. It is commonly used in credit risk modeling.

**Formula:**

```
Score = Offset + Factor * z
```

Where:
- z: The linear combination from the Logistic Regression model.
- Offset: Base score corresponding to the base odds.
- Factor: Scaling factor to adjust the score range.

**Example:**

Assume:
- Base odds = 50:1 (P(Good) = 0.98)
- Offset = 600, Factor = 20

For z = 1.61:

```
Score = 600 + 20 * 1.61 = 632
```

---

## Skor Kartı Formülü ve Lojistik Regresyon Bağlantısı

> **Skor kartı oluştururken, logit(p) yani ln(odds) yerine doğrudan lojistik regresyonun verdiği lineer kombinasyonu (z) kullanırız.**
>
> Lojistik regresyonun temel çıktısı:
>
> ```
> z = b0 + b1*x1 + b2*x2 + ... + bn*xn
> ```
>
> Buradaki z, aslında ln(odds) yani logit(p) ifadesidir.
>
> Skor kartı formülü ise:
>
> ```
> Skor = Offset + Faktör * z
> ```
>
> Yani, odds'un logaritması yerine doğrudan modelin verdiği z değeri kullanılır.
>
> - Offset: Temel skor (genellikle belirli bir odds veya puan için ayarlanır)
> - Faktör: Skorun ölçeğini belirler (örneğin, her bir birimlik z değişimi için kaç puan artacağı)
>
> **Özetle:**
> - Lojistik regresyonun çıktısı olan z, ln(odds) ile aynıdır.
> - Skor kartı oluştururken, bu z değeri doğrudan skor formülüne konur.
> - Böylece her müşteri için modelin tahmin ettiği risk, puan olarak kolayca yorumlanabilir.

---

## 3. Lojistik Regresyonda z (logit(p)) Nasıl Hesaplanır?

> **z değeri, modelin tahmin ettiği logit(p) yani ln(odds) değeridir.**
>
> z, lojistik regresyonun katsayıları (b0, b1, ..., bn) ile her bir gözlemin feature (özellik) değerlerinin çarpılıp toplanmasıyla elde edilir:
>
> ```
> z = b0 + b1*x1 + b2*x2 + ... + bn*xn
> ```
>
> Burada:
> - b0: Modelin sabit (intercept) katsayısıdır.
> - b1, ..., bn: Her bir özelliğin (feature) katsayısıdır.
> - x1, ..., xn: İlgili gözlemin feature değerleridir (genellikle WOE ile dönüştürülmüş değerler).
>
> Yani, her müşteri/gözlem için z değeri, o müşterinin özellikleri ve modelin katsayıları ile yukarıdaki formüle göre hesaplanır.

---

## Summary of the Workflow
1. **WOE Transformation**: Raw data is binned and transformed into WOE values.
2. **Logistic Regression**: WOE-transformed data is used to train a Logistic Regression model.
3. **Scorecard**: The Logistic Regression model is converted into a scorecard for practical use.

This workflow ensures interpretability and scalability, making it ideal for credit risk modeling and other binary classification problems.

---

# WOE -> Lojistik Regresyon -> Skor Kartı Süreci

> **Not:** Bu projede WOE hesaplanırken hedef değişken (y) için 0 değerleri "İyi" (Good), 1 değerleri ise "Kötü" (Bad) olarak kabul edilmektedir. Yani:
> - y = 0 → İyi (Good)
> - y = 1 → Kötü (Bad)
> 
> WOE ve IV hesaplamalarında Good/Bad oranı bu şekilde kullanılmaktadır.

Bu belge, ham verilerin bir skor kartına dönüştürülmesi sürecini açıklar. Süreç, Ağırlık Kanıtı (WOE) dönüşümü, Lojistik Regresyon modellemesi ve nihai skor kartı oluşturmayı içerir. Her adım, formüller ve örneklerle detaylandırılmıştır.

---

## 1. Ağırlık Kanıtı (WOE)

**WOE Nedir?**  
Ağırlık Kanıtı (WOE), kategorik veya sürekli değişkenleri Lojistik Regresyon için uygun bir formata dönüştürmek için kullanılan bir tekniktir. Bir özelliğin tahmin gücünü, iyi ve kötü sonuçların dağılımını karşılaştırarak ölçer.

**Formül:**

```
WOE = ln(%İyi / %Kötü)
```

Burada:
- %İyi: Bin içindeki iyi sonuçların oranı.
- %Kötü: Bin içindeki kötü sonuçların oranı.

**Örnek:**

| Bin  | İyi  | Kötü | %İyi  | %Kötü | WOE   |
|------|------|------|-------|-------|-------|
| 1    | 50   | 10   | 0.833 | 0.167 | 1.61  |
| 2    | 30   | 20   | 0.600 | 0.400 | 0.51  |
| 3    | 20   | 70   | 0.222 | 0.778 | -1.25 |

---

## 2. Lojistik Regresyon

**Lojistik Regresyon Nedir?**  
Lojistik Regresyon, ikili bir sonucun (ör. iyi vs. kötü) olasılığını modellemek için kullanılan bir istatistiksel yöntemdir. Sonucun log-odds'unu giriş özelliklerinin doğrusal bir kombinasyonu olarak tahmin eder.

**Formül:**

```
z = b0 + b1*x1 + b2*x2 + ... + bn*xn
```

Pozitif sonucun olasılığı:

```
P(İyi) = exp(z) / (1 + exp(z))
```

**Örnek:**

Denklemi kullanarak:

```
z = 0.0525 - 0.3891*özellik_1 + 0.0641*özellik_2 - 1.2422*özellik_3
```

Eğer özellik_1 = 0, özellik_2 = 0, özellik_3 = 1:

```
z = 0.0525 - 1.2422 = -1.1897
P(İyi) = exp(-1.1897) / (1 + exp(-1.1897)) = 0.234
```

---

## Lojistik Regresyon Modelinde logit(p) ve Odds

> **Lojistik Regresyon modelimizde, model çıktısı (p) bir gözlemin "1" yani "Kötü" (Bad) olma olasılığıdır.**
>
> logit(p) ifadesi, bu olasılığın odds'unun (olasılık/karşı olasılık) logaritmasıdır:
>
> ```
> logit(p) = ln(p / (1 - p))
> ```
>
> Burada:
> - p: Bir gözlemin "Kötü" (Bad, y=1) olma olasılığıdır.
> - (1-p): Aynı gözlemin "İyi" (Good, y=0) olma olasılığıdır.
> - p / (1-p): "Kötü" olma odds'u, yani bir gözlemin kötü olma ihtimalinin iyi olma ihtimaline oranı.
>
> Modelin tahmin ettiği değerler, bu odds'un logaritmasıdır. Yani model, her gözlem için "Kötü" olma odds'unu tahmin eder.

---

## 3. Skor Kartı Oluşturma

**Skor Kartı Nedir?**  
Bir skor kartı, Lojistik Regresyon modelinin log-odds'un kullanıcı dostu bir skora dönüştürüldüğü bir temsildir. Genellikle kredi riski modellemesinde kullanılır.

**Formül:**

```
Skor = Offset + Faktör * z
```

Burada:
- z: Lojistik Regresyon modelinden gelen lineer kombinasyon.
- Offset: Temel skor (genellikle belirli bir odds veya puan için ayarlanır)
- Faktör: Skorun ölçeğini belirler (örneğin, her bir birimlik z değişimi için kaç puan artacağı)

**Örnek:**

Varsayalım:
- Temel olasılık = 50:1 (P(İyi) = 0.98)
- Offset = 600, Faktör = 20

z = 1.61 için:

```
Skor = 600 + 20 * 1.61 = 632
```

---

## Skor Kartı Formülü ve Lojistik Regresyon Bağlantısı

> **Skor kartı oluştururken, logit(p) yani ln(odds) yerine doğrudan lojistik regresyonun verdiği lineer kombinasyonu (z) kullanırız.**
>
> Lojistik regresyonun temel çıktısı:
>
> ```
> z = b0 + b1*x1 + b2*x2 + ... + bn*xn
> ```
>
> Buradaki z, aslında ln(odds) yani logit(p) ifadesidir.
>
> Skor kartı formülü ise:
>
> ```
> Skor = Offset + Faktör * z
> ```
>
> Yani, odds'un logaritması yerine doğrudan modelin verdiği z değeri kullanılır.
>
> - Offset: Temel skor (genellikle belirli bir odds veya puan için ayarlanır)
> - Faktör: Skorun ölçeğini belirler (örneğin, her bir birimlik z değişimi için kaç puan artacağı)
>
> **Özetle:**
> - Lojistik regresyonun çıktısı olan z, ln(odds) ile aynıdır.
> - Skor kartı oluştururken, bu z değeri doğrudan skor formülüne konur.
> - Böylece her müşteri için modelin tahmin ettiği risk, puan olarak kolayca yorumlanabilir.

---

### Lojistik Regresyonda z (logit(p)) Nasıl Hesaplanır?

> **z değeri, modelin tahmin ettiği logit(p) yani ln(odds) değeridir.**
>
> z, lojistik regresyonun katsayıları (b0, b1, ..., bn) ile her bir gözlemin feature (özellik) değerlerinin çarpılıp toplanmasıyla elde edilir:
>
> ```
> z = b0 + b1*x1 + b2*x2 + ... + bn*xn
> ```
>
> Burada:
> - b0: Modelin sabit (intercept) katsayısıdır.
> - b1, ..., bn: Her bir özelliğin (feature) katsayısıdır.
> - x1, ..., xn: İlgili gözlemin feature değerleridir (genellikle WOE ile dönüştürülmüş değerler).
>
> Yani, her müşteri/gözlem için z değeri, o müşterinin özellikleri ve modelin katsayıları ile yukarıdaki formüle göre hesaplanır.

---

## Sürecin Özeti
1. **WOE Dönüşümü**: Ham veriler binlere ayrılır ve WOE değerlerine dönüştürülür.
2. **Lojistik Regresyon**: WOE ile dönüştürülmüş veriler kullanılarak bir Lojistik Regresyon modeli eğitilir.
3. **Skor Kartı**: Lojistik Regresyon modeli, pratik kullanım için bir skor kartına dönüştürülür.

Bu süreç, yorumlanabilirlik ve ölçeklenebilirlik sağlar ve kredi riski modellemesi ve diğer ikili sınıflandırma problemleri için idealdir.