# VERI SETI: https://www.tcmb.gov.tr/wps/wcm/connect/TR/TCMB+TR/Main+Menu/Istatistikler/Enflasyon+Verileri/Tuketici+Fiyatlari
# EVDS API kullanilmistir

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import evds as ev
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from arch.unitroot import ADF,PhillipsPerron,KPSS,ZivotAndrews # birim kok testi

from scipy.stats import boxcox

from statsmodels.graphics.tsaplots import month_plot, plot_acf
import statsmodels.api as sm


from merlion.models.utils.autosarima_utils import nsdiffs


from sklearn.preprocessing import PolynomialFeatures
from scipy.special import inv_boxcox  # boxcox'i tersine çevirmek için


from sklearn.metrics import r2_score, mean_squared_error


from statsmodels.tsa.holtwinters import Holt, ExponentialSmoothing

from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch, acorr_breusch_godfrey

from statsmodels.tsa.x13 import x13_arima_select_order # normal order ve seasonal order yapilarini ortaya çikartir.
import os

from statsmodels.tsa.seasonal import seasonal_decompose

from scipy.stats import ttest_1samp, jarque_bera
from arch import arch_model

## iyilestirmeler - 13.12.24

from statsmodels.tsa.stattools import adfuller
import ruptures as rpt



api = "5Nca6TRSIR"
    
evds = ev.evdsAPI(api)

'''
tufe = evds.get_data(["TP.FG.J0"],startdate="01-01-2003",enddate="01-02-2025",frequency=5)
tufe.rename(columns={"TP_FG_J0":"TUFE"},inplace=True) # veri seti duzenleme
tarih=pd.date_range("01-01-2003",periods=len(tufe),freq="M") # tarih indekslemek için
tufe["Tarih"]=tarih
tufe.set_index("Tarih",inplace=True) 
'''

# doviz - dolar tl satis uzerinden ornek alindi

dolar = evds.get_data(["TP.DK.USD.A.YTL"], startdate="01-01-2003",enddate="01-02-2025",frequency=5)
dolar.rename(columns={"TP_DK_USD_A_YTL":"Dolar(TL)"},inplace=True)

tarih=pd.date_range("01-01-2003",periods=len(dolar),freq="M") # tarih indekslemek için
dolar["Tarih"]=tarih
dolar.set_index("Tarih",inplace=True)

# faiz - (ihtiyac kredisi uzerinden ornek alindi)
# faiz verisi: https://evds2.tcmb.gov.tr/index.php?/evds/portlet/K24NEG9DQ1s%3D/tr
ihtiyackredisi = evds.get_data(["TP.KTF10"], startdate="01-01-2003",enddate="01-02-2025",frequency=5)
ihtiyackredisi.rename(columns={"TP_KTF10":"Kredi(TL)"},inplace=True)

tarih=pd.date_range("01-01-2003",periods=len(ihtiyackredisi),freq="M")
ihtiyackredisi["Tarih"] = tarih
ihtiyackredisi.set_index("Tarih",inplace=True)



print(dolar.columns)
print('--')
print(ihtiyackredisi.columns)



'''
# Yani "tufe_train" veri setinde yer alan "Enflasyon", "Doviz_Kuru" ve "Faiz_Orani" gibi degişkenleri kullanacagiz.
# Doviz kuru ve faiz oranlarinin enflasyon uzerindeki etkisini modellemeyi amacliyor.

# Enflasyon verisi (y) ve dışsal değişkenler (x) (doviz kuru, faiz oranlari)
y = enf_test["Enf"]  # Enflasyon verisi
tufe_train["Doviz_Kuru"] = dolar["Dolar(TL)"]
tufe_train["Faiz_Orani"] = ihtiyackredisi["Kredi(TL)"]
x = tufe_train[["Doviz_Kuru", "Faiz_Orani"]]  # Dışsal değişkenler (petrol fiyatları çıkarıldı)
y, x = y.align(x, join="inner")

# SARIMAX modelini dışsal değişkenler ile kurarsak
model = SARIMAX(y, exog=x, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))  # Modeli oluşturuyoruz
results = model.fit() 

# Sonuçları inceleyelim
print(results.summary())

# Tahminler (test seti üzerinde)
y_pred = results.predict(start=len(tufe_train), end=len(tufe_train) + len(tufe_test) - 1,
                         exog=x)  # Dışsal değişkenleri test verisi ile kullanıyoruz

# Modelin tahminlerini görselleştirelim
plt.figure()
plt.plot(enf_test["Enflasyon"], label="Gerçek Değerler")  # Gerçek değerleri çiziyoruz
plt.plot(y_pred, label="Tahmin Edilen Değerler")  # Tahmin edilen değerleri çiziyoruz
plt.legend()
plt.title("Enflasyon Tahminleri: Döviz Kuru ve Faiz Oranı ile")
plt.show()
'''