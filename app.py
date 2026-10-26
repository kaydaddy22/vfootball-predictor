import io,re,math,datetime
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image,ImageEnhance,ImageOps
import pytesseract

st.set_page_config(page_title='V-Football Online',page_icon='⚽',layout='wide')
st.title('⚽ V-Football Predictor 4.0')
st.caption('Instant Virtuals • England • Odds-based analysis • Screenshots → review → new round')
st.warning('Market-derived estimates are not independently validated predictions. Virtual results may be random. No guaranteed wins or betting advantage.')
TEAMS=['FOR vs SUN','HUL vs CRY','LEE vs IPS','CHE vs NEW','TOT vs MUN','MCI vs COV','FUL vs BHA','LIV vs BRE','ARS vs AST','EVE vs BOU']
COLS=['Match','Home','Draw','Away','Over 2.5','Under 2.5','GG Yes','GG No']

def blank():
 return pd.DataFrame([{'Match':x,**{c:np.nan for c in COLS[1:]}} for x in TEAMS])
if 'round' not in st.session_state: st.session_state['round']=blank()
if 'history' not in st.session_state: st.session_state['history']=[]
if 'round_num' not in st.session_state: st.session_state['round_num']=0

def identify_market(s):
 low=s.lower()
 if 'gg/ng' in low or 'gg' in low and 'ng' in low or 'both teams' in low: return 'btts'
 if 'o/u' in low or 'over/under' in low or '2.5' in low: return 'ou'
 if '1x2' in low or '1 x 2' in low: return '1x2'
 return None

def scan(file):
 img=Image.open(file).convert('RGB')
 # upscale improves reading of phone screenshots
 img=ImageEnhance.Contrast(img).enhance(1.45)
 img=img.resize((img.width*2,img.height*2))
 raw=pytesseract.image_to_string(img,config='--psm 6')
 market=identify_market(raw)
 matches={}
 for line in raw.splitlines():
  norm=line.upper().replace('0/U','O/U').replace('VS','vs')
  m=re.search(r'\b([A-Z]{2,4})\s*vs\s*([A-Z]{2,4})\b',norm,re.I)
  if not m: continue
  key=f'{m.group(1).upper()} vs {m.group(2).upper()}'
  if key not in TEAMS: continue
  # Find odds after team names only; exclude 2.5 totals and non-price numbers
  rest=norm[m.end():]
  nums=[float(x.replace(',','.')) for x in re.findall(r'(?<!\d)(?:\d{1,2}[.,]\d{2})(?!\d)',rest)]
  nums=[v for v in nums if 1.01<=v<=99]
  if market=='1x2' and len(nums)>=3: matches[key]=nums[-3:]
  elif market in ('ou','btts') and len(nums)>=2: matches[key]=nums[-2:]
 return market,matches,raw

def fair(vals):
 vals=np.asarray(vals,dtype=float)
 if np.any(~np.isfinite(vals)) or np.any(vals<=1): return None
 inv=1/vals
 return inv/inv.sum()

def poisson_score_probs(h,d,a,ov=None,un=None,gg=None,ng=None):
 # Simple independent Poisson fit to available market probabilities; not a validated forecast.
 from scipy.optimize import minimize
 target=fair([h,d,a]); other=fair([ov,un]) if ov and un else None
 both=fair([gg,ng]) if gg and ng else None
 if target is None:return None
 def get(lh,la):
  k=np.arange(12)
  ph=np.exp(-lh)*np.power(lh,k)/np.array([math.factorial(int(i)) for i in k]); pa=np.exp(-la)*np.power(la,k)/np.array([math.factorial(int(i)) for i in k]); mat=np.outer(ph,pa);mat/=mat.sum()
  return mat
 def loss(z):
  lh,la=np.exp(z); mat=get(lh,la)
  p=np.array([np.tril(mat,-1).sum(),np.trace(mat),np.triu(mat,1).sum()])
  v=3*np.sum((p-target)**2)
  if other is not None:v+=2*(sum(mat[i,j] for i in range(12) for j in range(12) if i+j>=3)-other[0])**2
  if both is not None:v+=2*(mat[1:,1:].sum()-both[0])**2
  return v
 opt=minimize(loss,np.log([1.1,0.9]),bounds=[(-3,2),(-3,2)],method='L-BFGS-B')
 lh,la=np.exp(opt.x);mat=get(lh,la)
 scores=sorted([(f'{i}-{j}',mat[i,j]) for i in range(7) for j in range(7)],key=lambda x:-x[1])[:5]
 return lh,la,scores

st.subheader('1 · Upload all three screenshots')
files=st.file_uploader('Choose 1X2, O/U 2.5 and GG/NG screenshots together',type=['png','jpg','jpeg'],accept_multiple_files=True)
if st.button('📷 Read screenshots',type='primary',disabled=not files):
 candidate=blank(); notes=[]
 for f in files:
  try:
   market,matched,raw=scan(f)
   if not market:
    notes.append(f'⚠️ {f.name}: market not recognized.');continue
   fields={'1x2':['Home','Draw','Away'],'ou':['Over 2.5','Under 2.5'],'btts':['GG Yes','GG No']}[market]
   for name,odds in matched.items():
    for c,v in zip(fields,odds):candidate.loc[candidate['Match']==name,c]=v
   notes.append(f'{f.name}: {market}, {len(matched)} matches read')
  except Exception as e:notes.append(f'⚠️ {f.name}: {str(e)}')
 st.session_state['pending']=candidate
 st.session_state['scan_notes']=notes
if 'scan_notes' in st.session_state:
 for note in st.session_state['scan_notes']:st.write(note)

st.subheader('2 · Review extracted odds')
st.caption('OCR can misread prices. Correct missing/wrong numbers before applying. Blank markets are not guessed.')
pending=st.session_state.get('pending',st.session_state['round'])
edited=st.data_editor(pending,key='odds_editor',hide_index=True,use_container_width=True,num_rows='fixed',disabled=['Match'],column_config={c:st.column_config.NumberColumn(c,min_value=1.01,max_value=1000,format='%.2f') for c in COLS[1:]})
if st.button('✅ Apply reviewed odds • Replace previous round'):
 st.session_state['round']=edited.copy();st.session_state['round_num']+=1
 st.session_state.pop('pending',None);st.session_state.pop('scan_notes',None)
 st.success(f'Round {st.session_state["round_num"]} applied. Predictions below now use the new odds.')
 st.rerun()

st.subheader('3 · Full-round analysis')
filter_70=st.checkbox('Highlight outcomes with market-implied probabilities of 70–80%')
df=st.session_state['round'];rows=[]
for _,r in df.iterrows():
 x={'Match':r['Match']}
 for names,labels in [(['Home','Draw','Away'],['1','X','2']),(['Over 2.5','Under 2.5'],['Over 2.5','Under 2.5']),(['GG Yes','GG No'],['GG','NG'])]:
  p=fair([r[n] for n in names])
  if p is not None:
   for lab,prob in zip(labels,p):x[lab]=round(prob*100,1)
 rows.append(x)
view=pd.DataFrame(rows)
if filter_70:
 choices=[]
 for _,r in view.iterrows():
  for c in view.columns[1:]:
   if pd.notna(r.get(c)) and 70<=r[c]<=80:choices.append({'Match':r['Match'],'Market':c,'Implied probability':f'{r[c]:.1f}%'})
 st.dataframe(pd.DataFrame(choices),hide_index=True,use_container_width=True)
else:st.dataframe(view,hide_index=True,use_container_width=True)
st.caption('These probabilities normalize the bookmaker prices to remove the margin. They are NOT independent value estimates.')

st.subheader('4 · Individual match breakdown')
match=st.selectbox('Choose a match',df['Match'].tolist())
r=df.loc[df['Match']==match].iloc[0]
for names,label in [(['Home','Draw','Away'],'1X2'),(['Over 2.5','Under 2.5'],'Over / Under 2.5'),(['GG Yes','GG No'],'GG / NG')]:
 p=fair([r[n] for n in names]);
 if p is not None:st.write(f'**{label}:** '+ '  •  '.join(f'{n}: {v:.1%}' for n,v in zip(names,p)))
if all(pd.notna(r[c]) for c in ['Home','Draw','Away']):
 p=poisson_score_probs(r['Home'],r['Draw'],r['Away'],r['Over 2.5'] if pd.notna(r['Over 2.5']) else None,r['Under 2.5'] if pd.notna(r['Under 2.5']) else None,r['GG Yes'] if pd.notna(r['GG Yes']) else None,r['GG No'] if pd.notna(r['GG No']) else None)
 if p:
  st.write(f'**Illustrative Poisson fit:** expected goals {p[0]:.2f} – {p[1]:.2f}')
  st.write('**Modelled scorelines:** '+', '.join(f'{score} ({pr:.1%})' for score,pr in p[2]))
  st.caption('Poisson estimates are fitted to prices and can conflict with real virtual scoring behavior.')

st.subheader('5 · Paper bankroll simulator')
start=st.number_input('Starting virtual bankroll (₦)',min_value=100.,value=10000.,step=1000.)
odds=st.number_input('Example decimal odds',min_value=1.01,value=1.50,step=0.05)
winrate=st.slider('Assumed true win probability (unverified)',0,100,70)/100
stake=st.slider('Stake as % of bankroll',1,100,5)/100
bets=st.slider('Number of simulated bets',5,200,30)
if st.button('Simulate 1,000 possible paths'):
 rng=np.random.default_rng(42);bal=np.full(1000,start)
 for _ in range(bets):bal=np.where(rng.random(1000)<winrate,bal*(1+stake*(odds-1)),bal*(1-stake))
 st.write(f'Median final balance: **₦{np.median(bal):,.0f}** · Probability of reaching ₦100,000: **{np.mean(bal>=100000):.1%}** · Probability of finishing below start: **{np.mean(bal<start):.1%}**')
 st.caption('Illustrative only; results depend heavily on the assumed win rate, which has not been established.')

st.subheader('6 · Save / download your data')
st.download_button('⬇️ Download current round CSV',df.to_csv(index=False).encode(),file_name='vfootball_current_round.csv',mime='text/csv')
st.caption('On free hosting, server sessions may reset. Download the CSV if you want to preserve the odds.')
