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

tufe = evds.get_data(["TP.FG.J0"],startdate="01-01-2003",enddate="01-02-2025",frequency=5)
tufe.rename(columns={"TP_FG_J0":"TUFE"},inplace=True) # veri seti duzenleme



tarih=pd.date_range("01-01-2003",periods=len(tufe),freq="M") # tarih indekslemek için
tufe["Tarih"]=tarih
tufe.set_index("Tarih",inplace=True) # aylik indeksleme yaptik - evds api freq 5 => aylik


'''
df = tufe.resample("Y").mean() # yillik ortalama tufe (tuketici fiyatlari)
df.index = df.index.year


sns.heatmap(df,annot=True,fmt=".2f",cmap="Reds")
plt.show()

# tufe degerlerinden enflasyon hesaplamasi yapilcak ( veri islenecek )


tufe["Enflasyon"] = tufe.pct_change()*100 # aylik enflasyon
print(tufe)
'''

enfay = tufe.pct_change() * 100
enfay.dropna(inplace=True)
enfay = pd.DataFrame(enfay)
enfay.rename(columns={"TUFE":"Enf"},inplace=True)

'''
plt.plot(tufe)
plt.show()
'''


adf = ADF(tufe,trend="ct")
pp = PhillipsPerron(tufe,trend="ct")
kpss = KPSS(tufe,trend="ct")

#print(adf) # regresyon analizi yapiyor - test statistic - p value ve lags degerleri cikiyor
#print(pp)
#print(kpss)


# p value -> h0 testi 1 veya 0 || 0 ise h0 reddedilir duragandir.



###################### 13.12.24 İyilestirme

# Duraganlık Testi İcin Fonksiyonlar ADF ve PP

def adf_test(series):
    """
    ADF testi uygular ve sonucu döndürür.
    """
    result = adfuller(series, autolag="AIC")
    return {
        "Test Statistiği": result[0],
        "p-değeri": result[1],
        "Kritik Değerler": result[4],
        "Duragan mı?": "Evet" if result[1] < 0.05 else "Hayır"
    }

def pp_test(series):
    """
    Phillips-Perron testi uygular ve sonucu döndürür.
    """
    result = PhillipsPerron(series)
    return {
        "Test Statistiği": result.stat,
        "p-değeri": result.pvalue,
        "Duragan mı?": "Evet" if result.pvalue < 0.05 else "Hayır"
    }


##

## duraganlık icin uygulama
def duraganlik_uygula(data):
    """
    ADF ve PP testlerini uygular ve sonuçları bir DataFrame'de toplar.
    """
    adf_results = adf_test(data)
    pp_results = pp_test(data)
    
    results = pd.DataFrame({
        "Test": ["ADF", "PP"],
        "Test Statistiği": [
            adf_results["Test Statistiği"],
            pp_results["Test Statistiği"]
        ],
        "p-değeri": [
            adf_results["p-değeri"],
            pp_results["p-değeri"]
        ],
        "Duragan mı?": [
            adf_results["Duragan mı?"],
            pp_results["Duragan mı?"]
        ]
    })
    return results

# Testleri çalıştırma
duraganlik_sonuclari = duraganlik_uygula(tufe["TUFE"])
print(duraganlik_sonuclari)


####### Yapısal Kırılma Analizi - Ruptures Kutuphanesi


def yapisal_kirilma_testi(series):
    """
    Yapısal kırılma analizi yapar ve kırılma noktalarını döndürür.
    """
    algo = rpt.Binseg(model="l2").fit(series.values)
    breakpoints = algo.predict(n_bkps=5)
    return breakpoints

def plot_structural_breaks(series, breakpoints):
    """
    Yapısal kırılmaları görselleştirir plotlar.
    """
    rpt.display(series.values, breakpoints)
    plt.title("Yapısal Kırılmalar")
    plt.show()

# Kırılma analizi ve görselleştirme
breakpoints = yapisal_kirilma_testi(tufe["TUFE"])
plot_structural_breaks(tufe["TUFE"], breakpoints)


############# 13.12.24 iyilestirme sonu ############################





####### yapisal kirilma testi
# c - sabit - ct sabit trendi - t direkt trend

#print('yapisal kirilma')

trend = ["c","ct","t"] # yapisal kirilma olabilir mi ?

for i in trend:
    z = ZivotAndrews(tufe,trend=i)
    #print(z)

# h0 'i reddedemedigimiz icin (duragan bulamadigimiz icin) yapisal kirilma (anlamli) yok. - p value 1 ciktigindan h1 reddedilir.

tufe_train = tufe.iloc[:36]
tufe_test = tufe.iloc[-36:]
#print(tufe_test)


# boxcox

tufe_train_bx,lm = boxcox(tufe_train["TUFE"])
tufe_train["Box-Cox"] = tufe_train_bx

'''
print(tufe_train)
# regresyon modelinde deterministik trend anlamli cikti.
'''

# - - - -

# GUNCELLEME 15.12.24

# Dağılım görselleştirme
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
sns.histplot(tufe["TUFE"], kde=True, ax=ax[0])
ax[0].set_title("Box-Cox Öncesi Dağılım")
sns.histplot(tufe_train["Box-Cox"], kde=True, ax=ax[1])
ax[1].set_title("Box-Cox Sonrası Dağılım")
plt.tight_layout()
plt.show()

## GUNCELLEME SONU

tufe_train["Aylar"] = tufe_train.index.month
dummy=pd.get_dummies(tufe_train["Aylar"],prefix="Aylar",dtype=int,drop_first=False) #kukla degiskenlere çevrilerek regresyon modeli olusturulur. Aylar degeri 2 ise Aylar_2 gibi sutun degiskeni olusturur. Kolaylik saglar.
tufe_train=pd.concat([tufe_train,dummy],axis=1)

########## stats models

x = tufe_train.drop(columns=["TUFE","Box-Cox","Aylar"])
y = tufe_train["TUFE"]
model = sm.OLS(y,x).fit() # stat models regression linear model OLS
# print(model.summary())
# mevsimsellik bir deterministik yok,




########## stokastik mevsimsellik kontrol - merlion

a = nsdiffs(tufe_train["Box-Cox"],m=12,test="ch")
b = nsdiffs(tufe_train["Box-Cox"],m=12,test="ocsb")

#print(a)
#print(b)



########## indeksleme | regresyon modelleme yapabilmek için

indeks = np.arange(1,len(tufe_train)+1)
tufe_train["T"] = indeks


x = sm.add_constant(tufe_train["T"])
y = tufe_train["Box-Cox"]

# model testing
model = sm.OLS(y,x).fit()
# print(model.summary())


pol = PolynomialFeatures(degree=3)
xp = pol.fit_transform(tufe_train["T"].values.reshape(-1,1))


# model testing
model = sm.OLS(y,xp).fit()
# print(model.summary())
# Durbin Watson degeri artmasina ragmen p degeri artti



#
girdi = np.arange(len(tufe_train.index)+1,len(tufe_train.index)+37)
tahmin = model.predict(pol.fit_transform(girdi.reshape(-1, 1)))
tahmin = inv_boxcox(tahmin,lm) # yukarda kullandigimiz lambda degerini gonderdik
tufe_test["Tahmin"] = tahmin   # duragan olmadigi için sahte regresyonlarla çok verimli islem yapamiyoruz.
# print(tufe_test) # TUFE degerleriyle Tahmin degerlerimiz arasinda ciddi fark var - anlamsiz.

# print(r2_score(tufe_test["TUFE"],tufe_test["Tahmin"]))  # r2 score degeri -5.23 geldi
# egitimde %99 luk basari olmasina ragmen testlerde durum kotu. 
# duraganlik oldugunu gosteriyor.
# r2 yapisi sahte regresyon oldugunu anlayabiliyoruz.



############################ USTEL DUZELTMELER

##### HOLT MODEL
# HOLT model RMSE = 1468.83




model = Holt(tufe_train["Box-Cox"]).fit(optimized=True) 
tahmin = model.forecast(36) # 36 test verisi
tahmin = inv_boxcox(tahmin,lm) # boxcox donusumu

# print(r2_score(tufe_test["TUFE"],tahmin))  # -5.2680
# print(np.sqrt(mean_squared_error(tufe_test["TUFE"],tahmin))) # HOLT model RMSE = 1468.83



##### HOLT WINTERS MODEL
# HOLT WINTERS model RMSE min 1471 çikti.


# çarpimsal ve toplamsal olarak tiplere bakilacak
# seride 0 deger varsa sifirlayacagindan normalde sikinti çikartabilen bir model.


trend_tip = ["add","mul"]
seas_tip = ["add","mul"]
per = range(2,13) # periyot - mevsimsellik kalibi


sonuc = pd.DataFrame(columns=["Trend","Mevsimsellik","Periyod","R2","RMSE"])

for i in trend_tip:
    for j in seas_tip:
        for k in per:
            model = ExponentialSmoothing(tufe_train["Box-Cox"],trend=i,seasonal=j,seasonal_periods=k).fit(optimized=True)
            tahmin = model.forecast(36) # 36 veri için
            tahmin = inv_boxcox(tahmin,lm)
            rmse = np.sqrt(mean_squared_error(tufe_test["TUFE"],tahmin))
            sonuc = sonuc._append({"Trend":i,"Mevsimsellik":j,"Periyod":k,"RMSE:":rmse},ignore_index=True)

sonuc = sonuc.sort_values(by="RMSE") # sutun adina gore df'yi sortla
# print(sonuc)  minimum RMSE 1471 çikti.




##### ARIMA

model = auto_arima(tufe_train["Box-Cox"],seasonal=False,trace=False)
# print(model.summary())
# AIC -> -130
# SARIMAX(2, 1, 1) kullanildi

# seasonal yapisi kullanilarak

sonuc = pd.DataFrame(columns=["m","AIC"])

for i in range(2,13):
    model = auto_arima(tufe_train["Box-Cox"],seasonal=True,trace=False,m=i)
    sonuc = sonuc._append({"m":i,"AIC":model.aic()},ignore_index=True)

sonuc = sonuc.sort_values(by="AIC")
# print(sonuc)




##### SARIMAX

model = SARIMAX(tufe_train["Box-Cox"],order=(0,1,1),seasonal_order=(2,0,1,12),trend="c").fit()
# ma.l1 0 ise problem yok, 0'dan buyuk degerler icin problemdir
# order parametresinde buna dikkat edicez
# model summary kisminda Model:SARIMAX(0,1,2)x(2,..) gibi devam eden kisimda
# L1'de sikinti olmadigi için SARIMAX'in 0,1 degerlerini aldik.
# 2'yi 1'e indirdik ma.l2 1'den buyuk sikintili oldugu icin
# z ve P>|z| degerlerine bakarak cevap verdik.
# intercept yapisinda oldugundan trend="c" (constant-sabit) parametresi ekledik


# print(model.summary())
# AIC -> -120
# SARIMAX(0, 1, 1)



################# ARIMA KULLANILACAK
# oto korelasyon olmayacak
# arch etkisi olmayacak

# q model
q = acorr_ljungbox(model.resid)
h = het_arch(model.resid,nlags=1,ddof=3)[1] # modeli egitirken 3 parametre kullandigimiz için
# serbestlik derecesi ddof 3 oldu.

# print(q)
# print(h)



####### seasonal order kullanimi

# model = SARIMAX(tufe_train["Box-Cox"],order = (0,1,1), seasonal_order=(1,0,1,12),trend="n").fit()

# path = r'C:\WinX13\x13as'
# arima = x13_arima_select_order(tufe_train["Box-Cox"],x12path = path, outlier=True)
# outlier - aykiri gozlem varsa dikkate alir

# print(arima.order) # order (1,1,0)
# print(arima.sorder)  # seasonal order (0,1,1,12)



#########

enfay = tufe["TUFE"].pct_change() * 100
enfay.dropna(inplace=True)
enfay = pd.DataFrame(enfay)
enfay.rename(columns={"TUFE":"Enf"},inplace = True)

enf_train = enfay.iloc[:-36]
enf_test = enfay.iloc[-36:]

# mevsimsel duzeltme - seasonal decompose
ayris = seasonal_decompose(enf_train["Enf"],model="add", period = 12, extrapolate_trend = "freq")
# ayris.plot()
# plt.show()

veri = pd.concat([
    ayris.observed,
    ayris.trend,
    ayris.seasonal,
    ayris.observed - ayris.seasonal, # toplamsal model kullandigimiz icin / yerine çikartma yaptik
    ayris.resid],axis=1)        # resid -> rassal


veri.columns = ["Enf","Trend","Mevsimsellik","Mevsimsel Duzeltme","Rassal"]

# y = mevsimsellik + rassal + trend -> Enflasyon

# plt.plot(veri["Mevsimsel Duzeltme"])
# plt.show()

# mevsimsel duzeltmenin dogrusallikla bir korelasyonu var mi ona bakicaz

indeks = np.arange(1,len(veri)+1) # dogrusallik yapisi (girdi olarak kullanilacakti - linear regression)
# print(indeks)
# plt.plot(indeks)
# plt.show()

# yukari yonluyse (hatasi dusukse) yuksek r'2 degeri çikar ve dogrusal korelasyon vardir.
# numpy korelasyon matrisini kullanalim

# print(np.corrcoef(veri["Mevsimsel Duzeltme"],indeks))
# 0.21 'lik korelasyon çikti. -(linear degil)
# verimizde deterministik bir mevsimsellik var
# mevsimselligi dislayip mevsimsel duzeltme katsayisina ulastik.
# mevsimsel duzeltmeyi dogru modelleyebilirsek deterministik oldugu için ekleme yapip tahmin sonucu elde edebiliriz.
# mevsimsellik yapisinda trend yapisi olmadigi için dogrusal model yapisi kuramadik - kuramayiz.

# mevsimsel olmayan arima modeliyle devam edicez

model = auto_arima(veri["Mevsimsel Duzeltme"],seasonal = False, Trace = False)
# print(model.summary()) 
# AIC 458
# arch etkisi - (geriye kalan artiklar uzerinden kontrol edicez) ve otokorelasyon var mi ?


##### oto korelasyon
# ac = acorr_ljungbox(model.resid()) # rassal degerler

### arch etkisi
# h = het_arch(model.resid(), nlags = 1 , ddof=3)[1] # serbestlik derecesi ddof = 3 ( p + d )

# print(ac)  # anlamli korelasyon yok
# print(h) # 0.001 arch etkisi var.

enf_test["Mevsimsellik"] = veri["Mevsimsellik"].iloc[-36:].values
tahmin = model.predict(36)
enf_test["Tahmin"] = tahmin + enf_test["Mevsimsellik"]
# print(enf_test)
# enflasyonla tahmin degerleri arasinda çok fark var
# bunun nedeni arch etkisinden dolayi,

## mevsimsel duzeltmeyi kullanmicaz, sarima kullanicaz.

'''
for i in range(2,13):
    model = auto_arima(veri["Enf"],seasonal=True,m=i,trace=False)
    print("m:{} Aic: {}".format(i,model.aic()))
'''

'''
m:2 Aic: 515.2326035115791
m:3 Aic: 509.77785147828354
m:4 Aic: 502.2720704046204
m:5 Aic: 541.3718499213408
m:6 Aic: 509.2182209465135
m:7 Aic: 541.3718499213408
m:8 Aic: 518.7257272095954
m:9 Aic: 528.3586976084922
m:10 Aic: 541.3718499213408
m:11 Aic: 541.3718499213408
m:12 Aic: 494.3792266831782

Bu modelde Aic degeri oncekine gore yuksek çikti, kullanmayacagiz.
'''


# Yumusatma ustel Modelleri Deneyelim:
    
sonuc = pd.DataFrame(columns=["Donem", "RMSE"])
period = range(2,13)

# hold winters mevsimselligi de baz aliyor

for i in period:
    model = ExponentialSmoothing(enf_train["Enf"],trend="add",seasonal="add",seasonal_periods=i).fit(optimized=True)
    tahmin = model.forecast(36) # 36 donemlik
    rmse = np.sqrt(mean_squared_error(enf_test["Enf"], tahmin))
    sonuc = sonuc._append({"Donem":i,"RMSE":rmse},ignore_index=True)
    
    
sonuc = sonuc.sort_values(by="RMSE") # kucukten buyuge
# print(sonuc)

'''
    Donem      RMSE
1     3.0  3.904438
7     9.0  3.912075
0     2.0  3.933386
8    10.0  3.934485
6     8.0  3.948194
9    11.0  3.949118
10   12.0  3.970932
4     6.0  3.997553
2     4.0  4.140709
3     5.0  4.206096
5     7.0  4.227467

Periyot olarak 3 donem kullandigimizda en dusuk RMSE degerine ulasabiliyoruz
Deneyelim:
'''


model = ExponentialSmoothing(enf_train["Enf"],trend="add",seasonal="add",seasonal_periods=3).fit(optimized=True)
tahmin = model.forecast(36) # 36 donemlik
enf_test["Tahmin"] = tahmin
# print(enf_test)
    
    
sonuc = sonuc.sort_values(by="RMSE")
# print(sonuc)

'''
Donem      RMSE
0     3.0  3.904438
19   10.0  3.904438
18    9.0  3.904438
17    8.0  3.904438
16    7.0  3.904438
15    6.0  3.904438
14    5.0  3.904438
13    4.0  3.904438
12    3.0  3.904438
11    2.0  3.904438
21   12.0  3.904438
20   11.0  3.904438
1     9.0  3.912075
2     2.0  3.933386
3    10.0  3.934485
4     8.0  3.948194
5    11.0  3.949118
6    12.0  3.970932
7     6.0  3.997553
8     4.0  4.140709
9     5.0  4.206096
10    7.0  4.227467
'''

# mevsimsel duzeltme = trend + rassal oldugundan rassal'lik enflasyona ciddi etkisi var.
# sabir bir ortalama modeliyle tahmin etmek zordur
# -----------------

### veri gorsellestirelim

veri.plot(y=["Enf","Trend","Rassal"])
#plt.show()

# enflasyon rassal soklarla hareket ediyor.
# trend veya rassali tek basina modelleyebilir miyiz ?

# trend -> belirli bir ortalama etrafinda gidiyor - linear regresyonla açiklanmaz
# mevsimsellik -> deterministiktir.
# rassallik -> bunun uzerinde durucaz.


indeks = np.arange(1,len(veri)+1)
veri["T"] = indeks

# polinomal regresyonla

pol = PolynomialFeatures(degree = 3)
xp = pol.fit_transform(veri["T"].values.reshape(-1,1))

y = veri["Trend"]
model = sm.OLS(y,xp).fit()
# print(model.summary())

# Log - Likehood = Katsayilar Anlamli
# Prob (F-statistic) > 0 : Model Anlamli
# R^2 dusuk çunku dogrusal model degil.

tahmin = model.predict(xp)

# sns.scatterplot(x = veri["T"], y=veri["Trend"], label="Gerçek Degerler")
# sns.lineplot(x = veri["T"], y=tahmin, label="Tahmin")
# plt.show()

#print(model.summary()) # polinom derecesi 1 'ken R^2 0.35
                       # polinom derecesi 2 'ken R^2 0.595
                       # polinom derecesi 3 'ken R^2 0.630


## modelin artiklari
artik = model.resid

## plot acf - otokoralasyon yapisi
#plot_acf(artik, zero=False)
#plt.show()


# acorr_breusch_godfrey - artiklarin otokoralasyonunu test edicek
h = acorr_breusch_godfrey(model)[1] # h1 - otokoralasyonun var oldugunu | h
# prop degeri onemli oldugundan modelin [1] degerini aldik.

# print(h) - 5.5162208818113e-40  - h1 kabul (h>0)


######### ARIMA ile Trendi tahmin etme
# trend modeli

model = auto_arima(veri["Trend"],seasonal = False, trace= False)
# print(model.summary())
# ar degerleri yok, (mevsimsellik kontrol edilmedi)

artik = model.resid()
ac = acorr_ljungbox(artik)
# print(ac) # otokoralasyon testi - anlamli degil.

# arch etkisi testi
h = het_arch(artik, nlags=1, ddof=3)[1]
# print(h) 0.9953081082332432 - p-degeri > 0.05 || 0.99 > 0.05 H0 kabul (arch etkisi yok)


#### tahmin
tahmin = model.predict_in_sample()
# print('TAHMIN:')
# print(tahmin)

# plt.plot(veri["Trend"],label="Trend")
# plt.plot(tahmin,label = "Tahmin")
# plt.legend()
# plt.show()



####### Rassal Hareket Analizi
## Ozetle
# Deterministik Mevsimsellik var
# Trend -> arch etkisi yok , otokorelasyon yapisi yok.
# Rassal'in Enflasyona etkisi en fazla derecede.
# ttest ile


# rassalligin ortalamasi sifir mi fark li mi ?

t_ist,p = ttest_1samp(veri["Rassal"], popmean=0)
# popmean -> hangi degerden farkli olup olmadigini test etmek icin verdigimiz deger
# print(p)
# 0.9255763888159445   -   h0 kabul arima yerine zero modeli secilebilir ( 0 ortalamaya sahip )


#### ADF augmented dickey fuller Testi - Rassal
# ortalama 0 oldugu icin trend n gonderebiliriz

adf = ADF(veri["Rassal"],trend="n")
# print(adf)

'''
Test Statistic: -8
Trend: No Trend
Critical Values: -2.58 (1%), -1.94 (5%), -1.62 (10%)
Null Hypothesis: The process contains a unit root.
'''


#### PhillipsPerron
pp = PhillipsPerron(veri["Rassal"],trend="n")
## print(pp)

'''
Test Statistic                -16.056
P-value                         0.000
Lags                               15
-------------------------------------

Trend: No Trend
Critical Values: -2.58 (1%), -1.94 (5%), -1.62 (10%)
Null Hypothesis: The process contains a unit root.
'''


###

sns.histplot(veri["Rassal"],kde=True)
# plt.show()
# kde - normal dagilim


##### normallik testi

n = jarque_bera(veri["Rassal"]) # normal dagilima uymamasini bekliyoruz..
# h0 normal dagilir
# h1 normal dagilmaz

# print(n)
# SignificanceResult(statistic=1076.5207353062297, pvalue=1.723822325527858e-234)
# normal dagilim yok
# seri duragan



#### ARCH ile modelleyecegiz

p_val = range(1,5) # 5 i kapsamayacagi icin max 4 gecikme denemesi
q_val = range(1,5)
o_val = range(0,2)

dag = ["ged","normal","studentst","skewt"]
model = ["GARCH","EGARCH"]

sonuc = pd.DataFrame(columns=["p","o","q","Model","Dagilim","Aic"])

for p in p_val:
    for q in q_val:
        for o in o_val:
            for d in dag:
                for m in model:
                    modelarch = arch_model(veri["Rassal"],p=p,o=o,q=q,vol=m,dist=d,mean="Zero").fit(disp="off")
                    sonuc = sonuc._append({"p":p,"o":o,"q":q,"Model":m,"Dagilim":d,"Aic":modelarch.aic},ignore_index=True)

sonuc = sonuc.sort_values(by="Aic")

# print(sonuc)

'''

p  o  q   Model Dagilim         Aic
14   1  1  1   GARCH   skewt  381.491804
30   1  1  2   GARCH   skewt  383.491803
78   2  1  1   GARCH   skewt  383.491804
142  3  1  1   GARCH   skewt  385.491803
46   1  1  3   GARCH   skewt  385.491803
..  .. .. ..     ...     ...         ...
114  2  0  4   GARCH  normal  413.718610
35   1  0  3  EGARCH  normal  414.308609
99   2  0  3  EGARCH  normal  415.158920
51   1  0  4  EGARCH  normal  416.308608
115  2  0  4  EGARCH  normal  417.158921

'''
# model-1 en dusuk AIC degerli;

modelarch = arch_model(veri["Rassal"],p=1,o=1,q=1,vol="GARCH",dist="skewt",mean="Zero").fit(disp="off")
# print(modelarch)
# ortalamasi zaten sifir

# arch etkisi kontrol 
h = het_arch(modelarch.std_resid, nlags=1)[1]
print(h)
# 0.859623524612982 -> H0 kabul arch etkisi yok.

plt.plot(veri["Rassal"])
plt.plot(modelarch.conditional_volatility) # aykiri degerleri ne kadar tahmin edebiliyoruz?

# ### OZET ####
# Mevsimsellik -> Deterministik
# Trend -> Arima ile modellenecek




# ##### Rolling Window - Kayan Yapi
test_sayisi = len(enf_test)
vol_tahmin = [] # volatility olmadigi icin.

for i in range(test_sayisi):
    train = enfay["Enf"][:-(test_sayisi-i)]
    modelarch = arch_model(train, p=1, o=1, q=1, vol="GARCH", dist="skewt", mean="Zero")
    model_arch = modelarch.fit(disp="off")
    tahmin = model_arch.forecast(horizon=1) # horizon -> kaç adim sonrasini tahmin edecek??
    vol_tahmin.append(np.sqrt(tahmin.variance.values[-1,:][0])) # Array Yapisindan çikarmak için ekstradan [0] parametresi ekledik.



# ## Trend Tahmin - ARIMA
model_trend = auto_arima(veri["Trend"],seasonal=False, trace=False)
tahmin_trend = model_trend.predict(test_sayisi)

tahmin_mev = veri["Mevsimsellik"].iloc[-(test_sayisi):]
# print(tahmin_mev)


# sonuc - toplamsal model
plt.figure()
son = tahmin_trend + tahmin_mev.values
enf_test["Tahmin"] = son

plt.ylim(top=13)
plt.plot(enf_train["Enf"], label = "Train")
plt.plot(enf_test["Enf"], label="Test")
plt.plot(enf_test["Tahmin"], label="Tahmin")
plt.legend()
plt.show()


# Model Belirleme SON
# ARIMA modelimiz volatility degerlerinde biraz zorlaniyor ama genel olarak en iyi accuracye sahip
plt.figure()
son = tahmin_trend + tahmin_mev.values + vol_tahmin
enf_test["Tahmin"] = son
plt.ylim(top=13)
plt.plot(enf_train["Enf"], label = "Train")
plt.plot(enf_test["Enf"], label="Test")
plt.plot(enf_test["Tahmin"], label="Tahmin")
plt.legend()
plt.show()



####### Doviz ve Faiz oranlarinin enflasyona etkisi analizi

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


train_size = int(len(enf_test) * 0.8)

train = enf_test.iloc[:train_size]
test = enf_test.iloc[train_size:]  

train["Doviz_Kuru"] = dolar["Dolar(TL)"]
train["Faiz_Orani"] = ihtiyackredisi["Kredi(TL)"]


y_train = train["Enf"]  
x_train = train["Doviz_Kuru"]   
x_train = train["Faiz_Orani"]


y_test = test["Enf"]
x_test = train["Doviz_Kuru"] 
x_test = train["Faiz_Orani"]


print("y_train boyutu:", len(y_train)) # 28
print("y_test boyutu:", len(y_test)) # 8
print("x_test boyutu:", len(x_test)) # 28 -> 8 reshalepeledik.


x_test = np.array(x_test[:8]).reshape(-1, 1)


model = SARIMAX(y_train, exog=x_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
results = model.fit()


y_pred = results.predict(start=len(y_train), 
                         end=len(y_train) + len(y_test) - 1, 
                         exog=x_test)


plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label="Gerçek Değerler", color='blue')  # Actual values
plt.plot(y_test.index, y_pred, label="Tahmin Edilen Değerler", color='red')  # Predictions
plt.legend()
plt.title("Enflasyon Tahminleri: Döviz Kuru ve Faiz Oranı ile")
plt.xlabel("Tarih")
plt.ylabel("Enflasyon")
plt.show()

