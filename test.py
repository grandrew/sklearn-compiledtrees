
from collections import namedtuple
from datetime import datetime
from functools import partial
from sklearn import ensemble, datasets
import argparse
import gc
import matplotlib.pyplot as plt
import numpy as np
import json
import time

plat = lambda: "PyPy"
import platform
platform.python_implementation = plat
from compiledtrees.compiled import CompiledClassifierPredictor

SAMPLES = 1000000

def evaluate_0(f, o):
  if f[7] <= -1.5183702111244202:
    if f[6] <= 1.2691466808319092:
      if f[2] <= 1.0930749773979187:
        if f[2] <= -1.1079156994819641:
          if f[2] <= -1.6593486666679382:
            o[0] += 0.00850574712643678
            o[1] += 0.32482758620689656
          
          else:
            o[0] += 0.05168220510741792
            o[1] += 0.28165112822591537
          
        
        else:
          if f[6] <= -1.2081618309020996:
            o[0] += 0.03513681592039801
            o[1] += 0.2981965174129353
          
          else:
            o[0] += 0.10413545901990157
            o[1] += 0.22919787431343175
          
        
      
      else:
        if f[7] <= -1.940596878528595:
          if f[6] <= 1.268014371395111:
            o[0] += 0.011213047910295615
            o[1] += 0.32212028542303767
          
          else:
            o[0] += 0.3333333333333333
            o[1] += 0.0
          
        
        else:
          if f[4] <= 1.020809531211853:
            o[0] += 0.054804804804804805
            o[1] += 0.27852852852852855
          
          else:
            o[0] += 0.01137357830271216
            o[1] += 0.32195975503062113
          
        
      
    
    else:
      if f[5] <= -0.8794727623462677:
        if f[7] <= -1.5196533799171448:
          if f[4] <= 0.3728974908590317:
            o[0] += 0.007001166861143524
            o[1] += 0.32633216647218977
          
          else:
            o[0] += 0.0
            o[1] += 0.3333333333333333
          
        
        else:
          o[0] += 0.3333333333333333
          o[1] += 0.0
        
      
      else:
        if f[4] <= 0.7780333459377289:
          if f[6] <= 1.7280970811843872:
            o[0] += 0.05025808204292312
            o[1] += 0.2830752512904102
          
          else:
            o[0] += 0.010077519379844961
            o[1] += 0.32325581395348835
          
        
        else:
          if f[6] <= 1.5179480910301208:
            o[0] += 0.020603384841795434
            o[1] += 0.3127299484915379
          
          else:
            o[0] += 0.0026666666666666666
            o[1] += 0.33066666666666666
          
        
      
    
  
  else:
    if f[7] <= 1.4091435074806213:
      if f[4] <= 1.5054141283035278:
        if f[2] <= -1.4671148657798767:
          if f[6] <= 1.2285898923873901:
            o[0] += 0.10210400533428399
            o[1] += 0.23122932799904933
          
          else:
            o[0] += 0.032817627754336616
            o[1] += 0.3005157055789967
          
        
        else:
          if f[0] <= 1.5611842274665833:
            o[0] += 0.20240999990925845
            o[1] += 0.13092333342407486
          
          else:
            o[0] += 0.08799841512941839
            o[1] += 0.24533491820391493
          
        
      
      else:
        if f[4] <= 2.0568422079086304:
          if f[9] <= -1.105224072933197:
            o[0] += 0.049328980776206025
            o[1] += 0.2840043525571273
          
          else:
            o[0] += 0.11820615011917321
            o[1] += 0.21512718321416013
          
        
        else:
          if f[6] <= -1.0351157784461975:
            o[0] += 0.005210368633580826
            o[1] += 0.3281229646997525
          
          else:
            o[0] += 0.035439172250330456
            o[1] += 0.2978941610830029
          
        
      
    
    else:
      if f[7] <= 1.9516347646713257:
        if f[2] <= 1.2692217826843262:
          if f[5] <= 1.4300230741500854:
            o[0] += 0.12242941661049951
            o[1] += 0.2109039167228338
          
          else:
            o[0] += 0.03083816552451239
            o[1] += 0.3024951678088209
          
        
        else:
          if f[1] <= 1.0279667973518372:
            o[0] += 0.044537521815008724
            o[1] += 0.2887958115183246
          
          else:
            o[0] += 0.006725514922236234
            o[1] += 0.3266078184110971
          
        
      
      else:
        if f[8] <= -0.9222736954689026:
          if f[8] <= -1.5067296624183655:
            o[0] += 0.0027258566978193145
            o[1] += 0.33060747663551404
          
          else:
            o[0] += 0.016654854712969524
            o[1] += 0.3166784786203638
          
        
        else:
          if f[4] <= -1.1337883472442627:
            o[0] += 0.0075403949730700175
            o[1] += 0.3257929383602633
          
          else:
            o[0] += 0.04512260176958593
            o[1] += 0.2882107315637474
          
        
      
    
  


def evaluate_1(f, o):
  if f[6] <= -1.4582456946372986:
    if f[9] <= 1.3278342485427856:
      if f[9] <= -1.0846794247627258:
        if f[9] <= -1.5688661336898804:
          if f[6] <= -1.9153331518173218:
            o[0] += 0.0010345541071798054
            o[1] += 0.3322987792261535
          
          else:
            o[0] += 0.022487792341300435
            o[1] += 0.3108455409920329
          
        
        else:
          if f[3] <= 1.3400554060935974:
            o[0] += 0.058861411643482735
            o[1] += 0.2744719216898506
          
          else:
            o[0] += 0.00530035335689046
            o[1] += 0.3280329799764429
          
        
      
      else:
        if f[8] <= 1.178552508354187:
          if f[6] <= -2.023699998855591:
            o[0] += 0.039461435415389504
            o[1] += 0.2938718979179438
          
          else:
            o[0] += 0.1270581751257295
            o[1] += 0.2062751582076038
          
        
        else:
          if f[0] <= 1.0196215510368347:
            o[0] += 0.04219333092572529
            o[1] += 0.291140002407608
          
          else:
            o[0] += 0.007898894154818325
            o[1] += 0.325434439178515
          
        
      
    
    else:
      if f[9] <= 1.6255384683609009:
        if f[4] <= -0.9548132419586182:
          if f[2] <= 0.47258980572223663:
            o[0] += 0.010718113612004287
            o[1] += 0.322615219721329
          
          else:
            o[0] += 0.0
            o[1] += 0.3333333333333333
          
        
        else:
          if f[8] <= 0.9573806524276733:
            o[0] += 0.055608667941363925
            o[1] += 0.2777246653919694
          
          else:
            o[0] += 0.013303769401330377
            o[1] += 0.32002956393200294
          
        
      
      else:
        if f[6] <= -1.460314929485321:
          if f[9] <= 1.836732804775238:
            o[0] += 0.018518518518518517
            o[1] += 0.3148148148148148
          
          else:
            o[0] += 0.00224372458280746
            o[1] += 0.3310896087505259
          
        
        else:
          if f[4] <= -0.5864515006542206:
            o[0] += 0.29166666666666663
            o[1] += 0.041666666666666664
          
          else:
            o[0] += 0.017543859649122806
            o[1] += 0.3157894736842105
          
        
      
    
  
  else:
    if f[7] <= -1.5151830315589905:
      if f[6] <= 1.098069667816162:
        if f[8] <= -1.341676652431488:
          if f[7] <= -1.871319830417633:
            o[0] += 0.007647907647907647
            o[1] += 0.32568542568542563
          
          else:
            o[0] += 0.04054904923813121
            o[1] += 0.29278428409520213
          
        
        else:
          if f[8] <= 1.3059510588645935:
            o[0] += 0.1014396666035234
            o[1] += 0.2318936667298099
          
          else:
            o[0] += 0.026707029936091486
            o[1] += 0.30662630339724184
          
        
      
      else:
        if f[6] <= 1.6540256142616272:
          if f[7] <= -1.852731704711914:
            o[0] += 0.01858952015801092
            o[1] += 0.3147438131753224
          
          else:
            o[0] += 0.066006600660066
            o[1] += 0.26732673267326734
          
        
        else:
          if f[2] <= 0.6476919651031494:
            o[0] += 0.01085203108960258
            o[1] += 0.3224813022437307
          
          else:
            o[0] += 0.0012062726176115801
            o[1] += 0.33212706071572173
          
        
      
    
    else:
      if f[0] <= 1.4730169773101807:
        if f[5] <= -1.4960533380508423:
          if f[5] <= -1.945493459701538:
            o[0] += 0.04175513092710544
            o[1] += 0.29157820240622784
          
          else:
            o[0] += 0.11966978100746453
            o[1] += 0.21366355232586878
          
        
        else:
          if f[3] <= 1.5035959482192993:
            o[0] += 0.2012065265915481
            o[1] += 0.13212680674178517
          
          else:
            o[0] += 0.09371107500676956
            o[1] += 0.23962225832656378
          
        
      
      else:
        if f[3] <= 1.2289664149284363:
          if f[0] <= 2.0531409978866577:
            o[0] += 0.11837308613686075
            o[1] += 0.21496024719647255
          
          else:
            o[0] += 0.03310026300596575
            o[1] += 0.30023307032736757
          
        
        else:
          if f[7] <= 0.9322735071182251:
            o[0] += 0.03565660330475835
            o[1] += 0.297676730028575
          
          else:
            o[0] += 0.008221752407798918
            o[1] += 0.3251115809255344
          
        
      
    
  


def evaluate_2(f, o):
  if f[1] <= -1.4810688495635986:
    if f[1] <= -2.029492497444153:
      if f[1] <= -2.340519428253174:
        if f[1] <= -2.5256142616271973:
          if f[1] <= -2.7338558435440063:
            o[0] += 0.0
            o[1] += 0.3333333333333333
          
          else:
            o[0] += 0.005874265716785402
            o[1] += 0.3274590676165479
          
        
        else:
          if f[5] <= -0.3674258887767792:
            o[0] += 0.007097405775819872
            o[1] += 0.32623592755751346
          
          else:
            o[0] += 0.02568154879494271
            o[1] += 0.3076517845383906
          
        
      
      else:
        if f[3] <= -1.0426114201545715:
          if f[2] <= 0.52513787150383:
            o[0] += 0.01633728590250329
            o[1] += 0.31699604743083004
          
          else:
            o[0] += 0.0012254901960784314
            o[1] += 0.3321078431372549
          
        
        else:
          if f[5] <= 1.2058499455451965:
            o[0] += 0.05766318340148417
            o[1] += 0.2756701499318492
          
          else:
            o[0] += 0.007387706855791961
            o[1] += 0.32594562647754133
          
        
      
    
    else:
      if f[9] <= 1.2461469173431396:
        if f[3] <= 1.208640456199646:
          if f[4] <= 1.2165840864181519:
            o[0] += 0.12552923337468735
            o[1] += 0.20780409995864596
          
          else:
            o[0] += 0.04433458882611425
            o[1] += 0.2889987445072191
          
        
        else:
          if f[6] <= 0.7912721931934357:
            o[0] += 0.04610299234516353
            o[1] += 0.28723034098816974
          
          else:
            o[0] += 0.011818181818181818
            o[1] += 0.3215151515151515
          
        
      
      else:
        if f[4] <= -0.9738139808177948:
          if f[9] <= 1.5776181817054749:
            o[0] += 0.014109347442680775
            o[1] += 0.31922398589065254
          
          else:
            o[0] += 0.0014398848092152627
            o[1] += 0.33189344852411806
          
        
        else:
          if f[3] <= -1.1417753100395203:
            o[0] += 0.005172413793103448
            o[1] += 0.32816091954022986
          
          else:
            o[0] += 0.04371294575701265
            o[1] += 0.2896203875763207
          
        
      
    
  
  else:
    if f[1] <= 1.4203215837478638:
      if f[0] <= 1.560437560081482:
        if f[6] <= 1.490976631641388:
          if f[6] <= -1.4505481123924255:
            o[0] += 0.09685573086688429
            o[1] += 0.23647760246644903
          
          else:
            o[0] += 0.2034346729884715
            o[1] += 0.1298986603448618
          
        
        else:
          if f[2] <= 1.2966914176940918:
            o[0] += 0.09699486438773873
            o[1] += 0.2363384689455946
          
          else:
            o[0] += 0.026692332370869425
            o[1] += 0.3066410009624639
          
        
      
      else:
        if f[3] <= 1.3513320684432983:
          if f[6] <= -1.2495388984680176:
            o[0] += 0.025635767022149304
            o[1] += 0.307697566311184
          
          else:
            o[0] += 0.09136065282951925
            o[1] += 0.24197268050381407
          
        
        else:
          if f[6] <= 1.1563329696655273:
            o[0] += 0.022866978116547823
            o[1] += 0.31046635521678545
          
          else:
            o[0] += 0.0010288065843621398
            o[1] += 0.33230452674897115
          
        
      
    
    else:
      if f[9] <= -1.1938738822937012:
        if f[4] <= 1.0610316395759583:
          if f[6] <= -1.1340453624725342:
            o[0] += 0.0064412238325281795
            o[1] += 0.32689210950080516
          
          else:
            o[0] += 0.04060239184999262
            o[1] += 0.29273094148334067
          
        
        else:
          if f[1] <= 1.730760633945465:
            o[0] += 0.014814814814814814
            o[1] += 0.3185185185185185
          
          else:
            o[0] += 0.0004432624113475177
            o[1] += 0.3328900709219858
          
        
      
      else:
        if f[2] <= -1.1201790571212769:
          if f[9] <= 1.1397041082382202:
            o[0] += 0.04582717376696681
            o[1] += 0.2875061595663665
          
          else:
            o[0] += 0.008086934546373515
            o[1] += 0.3252463987869598
          
        
        else:
          if f[1] <= 1.888941764831543:
            o[0] += 0.12844589550560787
            o[1] += 0.20488743782772545
          
          else:
            o[0] += 0.04930168434933764
            o[1] += 0.2840316489839957
          

def evaluate(f, probas):
  evaluate_0(f, probas)
  evaluate_1(f, probas)
  evaluate_2(f, probas)


d = datasets.make_hastie_10_2(n_samples=SAMPLES)
clf = ensemble.RandomForestClassifier(n_estimators=3, max_depth=5)
X, y = d
clf.fit(X, y)
# print("sklearn-trees:")
# ts = time.time()
# for d in X:
#     res = clf.predict_proba([d])
# print(time.time()-ts)
# print(res)
cclf = CompiledClassifierPredictor(clf)
X2 = X.tolist()
print("hyperc-trees:")
ts = time.time()
for d in X:
    res = cclf.predict_proba([d])
print(time.time()-ts)
print((time.time()-ts)/(SAMPLES/1000))
print(res)

n_samples = len(X)
n_features = len(X[0])
all_probas = [[0.0] * 2] * n_samples
probas2 = [0.0, 0.0]
print("hyperc-trees:")
ts = time.time()
for d in X:
    evaluate(d, probas2)
print(time.time()-ts)
print((time.time()-ts)/(SAMPLES/1000))
print(res)