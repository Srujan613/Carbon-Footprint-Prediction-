// ═══════════════════════════════════════════════
// CONFIG
// ═══════════════════════════════════════════════
const API_BASE_URL = window.CARBONSENSE_API_URL || 'http://localhost:5000';

// ═══════════════════════════════════════════════
// HERO CANVAS — rising CO₂ particles (index.html only)
// ═══════════════════════════════════════════════
const canvas = document.getElementById('hero-canvas');

if (canvas) {
  const ctx = canvas.getContext('2d');
  let ptcls = [];

  function resizeCanvas(){
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }

  class Particle{
    constructor(){this.reset(true);}
    reset(init){
      this.x = Math.random()*canvas.width;
      this.y = init ? Math.random()*canvas.height : canvas.height + 10;
      this.r = Math.random()*1.8+0.4;
      this.vx = (Math.random()-.5)*.35;
      this.vy = -(Math.random()*.55+.18);
      this.a = Math.random()*.5+.15;
      this.life = 0;
      this.maxLife = Math.random()*350+180;
    }
    update(){
      this.x+=this.vx; this.y+=this.vy; this.life++;
      if(this.life>this.maxLife||this.y<-10) this.reset(false);
    }
    draw(){
      const t = Math.min(this.life/50,1)*Math.min((this.maxLife-this.life)/50,1);
      ctx.globalAlpha = this.a*t;
      ctx.fillStyle = this.r>1.5?'#C4FF4D':'#1DB87A';
      ctx.beginPath();ctx.arc(this.x,this.y,this.r,0,Math.PI*2);ctx.fill();
    }
  }

  function initPtcls(){ptcls=[];for(let i=0;i<90;i++)ptcls.push(new Particle());}

  function animCanvas(){
    ctx.clearRect(0,0,canvas.width,canvas.height);
    ctx.globalAlpha=.025;ctx.strokeStyle='#C4FF4D';ctx.lineWidth=.5;
    for(let i=0;i<ptcls.length;i++){
      for(let j=i+1;j<ptcls.length;j++){
        const dx=ptcls[i].x-ptcls[j].x,dy=ptcls[i].y-ptcls[j].y;
        const d=Math.sqrt(dx*dx+dy*dy);
        if(d<110){ctx.globalAlpha=.025*(1-d/110);ctx.beginPath();ctx.moveTo(ptcls[i].x,ptcls[i].y);ctx.lineTo(ptcls[j].x,ptcls[j].y);ctx.stroke();}
      }
    }
    ptcls.forEach(p=>{p.update();p.draw();});
    requestAnimationFrame(animCanvas);
  }
  window.addEventListener('resize',()=>{resizeCanvas();initPtcls();});
  resizeCanvas();initPtcls();animCanvas();
}

// ═══════════════════════════════════════════════
// FORM STEP LOGIC (predict.html only)
// ═══════════════════════════════════════════════
let step = 1;
const TOTAL = 3;
let sexValue = 'male';
const titles = ['About You','Home & Energy','Transport'];
const subs   = ['Personal habits and dietary preferences','Household energy, appliances, and waste habits','Your commute, vehicle, and travel patterns'];

function goStep(dir){
  if(dir===1 && step===TOTAL){submitForm();return;}
  const next = step+dir;
  if(next<1||next>TOTAL) return;
  document.getElementById('fstep-'+step).classList.remove('active');
  step = next;
  document.getElementById('fstep-'+step).classList.add('active');
  for(let i=1;i<=TOTAL;i++){
    const p=document.getElementById('pill-'+i);
    p.classList.remove('active','done');
    if(i<step) p.classList.add('done');
    if(i===step) p.classList.add('active');
  }
  document.getElementById('fstep-title').textContent = titles[step-1];
  document.getElementById('fstep-sub').textContent = subs[step-1];
  document.getElementById('step-ctr').textContent = 'Step '+step+' of '+TOTAL;
  document.getElementById('btn-back').style.visibility = step===1?'hidden':'visible';
  document.getElementById('btn-next').textContent = step===TOTAL ? 'Calculate Footprint →' : 'Next →';
}

function setSex(btn, value){
  document.querySelectorAll('.tog-btn').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  sexValue = value;
  updateOrb();
}

function rng(id,displayId,fmt){
  document.getElementById(displayId).textContent = fmt(document.getElementById(id).value);
  updateOrb();
}

// ═══════════════════════════════════════════════
// PAYLOAD BUILDER — keys match the real dataset columns exactly
// (see backend/ml/features.py — keep these in sync with your training data)
// ═══════════════════════════════════════════════
function checkedValues(containerId){
  return Array.from(document.querySelectorAll(`#${containerId} input[type=checkbox]:checked`))
    .map(el => el.value);
}

function onTransportChange(){
  const transport = document.getElementById('transport').value;
  const vehicleSelect = document.getElementById('vehicle-type');
  const distInput = document.getElementById('vehicle-dist');
  if (transport !== 'private') {
    vehicleSelect.value = 'none';
    vehicleSelect.disabled = true;
    distInput.value = 0;
    distInput.disabled = true;
    rng('vehicle-dist', 'dist-val', v => parseInt(v).toLocaleString() + ' km');
  } else {
    vehicleSelect.disabled = false;
    distInput.disabled = false;
    if (vehicleSelect.value === 'none') vehicleSelect.value = 'petrol';
  }
  updateOrb();
}

function buildPayload(){
  const val = id => document.getElementById(id).value;
  return {
    "Body Type": val('body-type'),
    "Sex": sexValue,
    "Diet": val('diet'),
    "How Often Shower": val('shower'),
    "Heating Energy Source": val('heating'),
    "Transport": val('transport'),
    "Vehicle Type": val('vehicle-type'),
    "Social Activity": val('social'),
    "Monthly Grocery Bill": Number(val('grocery')),
    "Frequency of Traveling by Air": val('air-travel'),
    "Vehicle Monthly Distance Km": Number(val('vehicle-dist')),
    "Waste Bag Size": val('waste-size'),
    "Waste Bag Weekly Count": Number(val('waste-count')),
    "How Long TV PC Daily Hour": Number(val('tv-hours')),
    "How Many New Clothes Monthly": Number(val('new-clothes')),
    "How Long Internet Daily Hour": Number(val('net-hours')),
    "Energy efficiency": val('energy-eff'),
    "Recycling": checkedValues('recycling-group'),
    "Cooking_With": checkedValues('cooking-group'),
  };
}

// ═══════════════════════════════════════════════
// SUBMIT — calls the Flask backend for the real ensemble + SHAP result
// ═══════════════════════════════════════════════
async function submitForm(){
  const resultsEl = document.getElementById('results');
  resultsEl.style.display = 'block';
  setTimeout(()=>resultsEl.scrollIntoView({behavior:'smooth',block:'start'}), 150);

  setLoadingState(true);

  try {
    const res = await fetch(`${API_BASE_URL}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildPayload()),
    });

    if (!res.ok) {
      const errBody = await res.json().catch(()=>({}));
      throw new Error(errBody.error || `Request failed (${res.status})`);
    }

    const data = await res.json();
    renderResults(data);
  } catch (err) {
    renderError(err);
  } finally {
    setLoadingState(false);
  }
}

function setLoadingState(isLoading){
  const btn = document.getElementById('btn-next');
  if (step === TOTAL) {
    btn.disabled = isLoading;
    btn.textContent = isLoading ? 'Calculating…' : 'Calculate Footprint →';
  }
}

function renderError(err){
  document.getElementById('shap-items').innerHTML =
    `<p style="font-size:.8rem;color:var(--accent-coral);">Couldn't reach the prediction API (${err.message}). ` +
    `Make sure the Flask backend is running at ${API_BASE_URL} — see backend/README.md.</p>`;
}

function fmtNum(v){ return Math.round(v).toLocaleString(); }

function renderResults(data){
  const { predictions, shap, recommendations, context, mock_mode, archetype, stacking_available } = data;

  // Score hero
  document.getElementById('res-score').textContent = fmtNum(context.total_kg);
  document.getElementById('res-class').textContent = context.classification;
  document.getElementById('score-label').textContent = 'CatBoost Estimate (Best Model)';
  document.getElementById('res-context').textContent =
    `Sits ${context.total_kg >= context.world_avg_kg ? 'above' : 'below'} the global average of ` +
    `${fmtNum(context.world_avg_kg)} kg.` +
    (mock_mode ? ' (Demo estimate — connect your trained models for real predictions.)' : '');

  // Lifestyle archetype
  if (archetype && archetype.label) {
    document.getElementById('archetype-label').textContent = archetype.label;
    document.getElementById('archetype-note').textContent = archetype.cluster !== null && archetype.cluster !== undefined
      ? 'Assigned by the trained KMeans lifestyle-clustering model.'
      : 'Estimated (real KMeans archetype model not loaded on the backend).';
  }

  // SHAP feature importance
  const maxAbs = Math.max(...shap.map(s => Math.abs(s.value)), 1);
  document.getElementById('shap-items').innerHTML = shap.map(s => `
    <div class="shap-item">
      <div class="shap-hdr">
        <span class="shap-feat">${s.feature}</span>
        <span class="shap-v ${s.direction}">${s.direction === 'pos' ? '+' : '−'}${Math.abs(Math.round(s.value))}</span>
      </div>
      <div class="shap-track"><div class="shap-fill ${s.direction}" style="width:${Math.min(100, Math.abs(s.value)/maxAbs*50)}%"></div></div>
    </div>
  `).join('');

  // Model predictions
  const modelLabels = { ridge: 'Ridge', random_forest: 'Rand. Forest', catboost: 'CatBoost (Best Model)', lightgbm: 'LightGBM', xgboost: 'XGBoost' };
  const maxPred = Math.max(...Object.values(predictions));
  const rows = Object.entries(modelLabels).map(([key, label]) => {
    const isBest = key === 'catboost';
    return `
    <div class="mpred-row${isBest ? ' mpred-ensemble' : ''}">
      <span class="mpred-name"${isBest ? ' style="color:var(--accent-lime);font-weight:600;"' : ''}>${label}</span>
      <div class="mpred-track">
        <div class="mpred-fill" style="width:${(predictions[key]/maxPred*100).toFixed(0)}%;background:linear-gradient(90deg,${isBest ? 'var(--accent-lime),rgba(196,255,77,.6)' : 'rgba(196,255,77,.45),rgba(196,255,77,.2)'});">
          <span class="mpred-val"${isBest ? '' : ' style="color:var(--accent-lime)"'}>${fmtNum(predictions[key])}</span>
        </div>
      </div>
    </div>`;
  }).join('');
  const ensembleRow = `
    <hr class="divider" style="margin:10px 0;"/>
    <div class="mpred-row">
      <span class="mpred-name">Simple Average</span>
      <div class="mpred-track">
        <div class="mpred-fill" style="width:${(predictions.ensemble/maxPred*100).toFixed(0)}%;background:linear-gradient(90deg,rgba(196,255,77,.45),rgba(196,255,77,.2));">
          <span class="mpred-val" style="color:var(--accent-lime)">${fmtNum(predictions.ensemble)}</span>
        </div>
      </div>
    </div>`;
  const stackingRow = predictions.stacking !== undefined ? `
    <div class="mpred-row">
      <span class="mpred-name">Stacking Ensemble</span>
      <div class="mpred-track">
        <div class="mpred-fill" style="width:${(predictions.stacking/maxPred*100).toFixed(0)}%;background:linear-gradient(90deg,rgba(196,255,77,.45),rgba(196,255,77,.2));">
          <span class="mpred-val" style="color:var(--accent-lime)">${fmtNum(predictions.stacking)}</span>
        </div>
      </div>
    </div>` : '';
  document.getElementById('mpred-container').innerHTML = rows + ensembleRow + stackingRow;

  // Percentile
  document.getElementById('pct-num').innerHTML = `${context.percentile}<sup>th</sup>`;
  document.getElementById('pct-desc').textContent =
    `You emit more than ${context.percentile}% of people in the training dataset.`;
  setTimeout(()=>{ document.getElementById('pct-bar').style.width = context.percentile + '%'; }, 200);

  // Recommendations
  const priorityLabel = { high: '● High Impact', mid: '● Medium Impact', low: '● Quick Win' };
  document.getElementById('rec-grid').innerHTML = recommendations.length
    ? recommendations.map(r => `
      <div class="rec-card">
        <span class="rec-priority ${r.priority}">${priorityLabel[r.priority]}</span>
        <span class="rec-icon">${r.icon}</span>
        <h3 class="rec-title">${r.title}</h3>
        <p class="rec-desc">${r.desc}</p>
        <div class="rec-impact">
          <span class="rec-impact-label">Potential saving</span>
          <span class="rec-impact-val">↓ ${fmtNum(r.impact_kg)} kg CO₂/yr</span>
        </div>
      </div>
    `).join('')
    : `<p style="font-size:.85rem;color:var(--text-muted);">No specific hotspots found — your footprint is well balanced.</p>`;
}

// ═══════════════════════════════════════════════
// LIVE ESTIMATE — Carbon Pulse Orb (client-side preview while filling the form;
// the authoritative number comes from the backend on submit)
// ═══════════════════════════════════════════════
const dietMap = {omnivore:1500,pescatarian:900,vegetarian:580,vegan:280};
const heatMap = {coal:2900,'natural gas':1900,wood:1100,electricity:700,solar:150};
const airMap  = {never:0,rarely:480,frequently:1700,'very frequently':3800};

function getEstimate(){
  const diet     = document.getElementById('diet').value;
  const heating  = document.getElementById('heating').value;
  const air      = document.getElementById('air-travel').value;
  const dist     = parseInt(document.getElementById('vehicle-dist').value)||0;
  const grocery  = parseInt(document.getElementById('grocery').value)||200;
  const clothes  = parseInt(document.getElementById('new-clothes').value)||0;
  const tv       = parseInt(document.getElementById('tv-hours').value)||0;
  const net      = parseInt(document.getElementById('net-hours').value)||0;

  const dietScore      = dietMap[diet]||800;
  const transportScore = dist*0.19 + (airMap[air]||0);
  const homeScore      = (heatMap[heating]||1000) + grocery*0.45;
  const lifestyleScore = clothes*38 + tv*14 + net*7;

  return {
    total: Math.round(dietScore+transportScore+homeScore+lifestyleScore),
    transport: Math.round(transportScore),
    home: Math.round(homeScore),
    diet: Math.round(dietScore),
    lifestyle: Math.round(lifestyleScore)
  };
}

function fmtK(v){return v>=1000?(v/1000).toFixed(1)+'k':String(v);}

function updateOrb(){
  const e = getEstimate();
  const orbNum  = document.getElementById('orb-num');
  const badge   = document.getElementById('emission-badge');
  const core    = document.getElementById('orb-core');

  orbNum.textContent = fmtK(e.total);

  let levelText,bgGrad,borderColor,badgeBg,badgeColor;
  if(e.total<2000){
    levelText='Low emissions';
    bgGrad='radial-gradient(circle at 40% 38%,rgba(29,184,122,.32),rgba(29,184,122,.1),transparent)';
    borderColor='rgba(29,184,122,.35)';badgeBg='rgba(29,184,122,.1)';badgeColor='#1DB87A';
    orbNum.style.color='#1DB87A';
  } else if(e.total<5000){
    levelText='Around average';
    bgGrad='radial-gradient(circle at 40% 38%,rgba(196,255,77,.25),rgba(29,184,122,.1),transparent)';
    borderColor='rgba(196,255,77,.28)';badgeBg='rgba(196,255,77,.08)';badgeColor='#C4FF4D';
    orbNum.style.color='#C4FF4D';
  } else if(e.total<9000){
    levelText='Above average';
    bgGrad='radial-gradient(circle at 40% 38%,rgba(255,184,48,.28),rgba(196,255,77,.1),transparent)';
    borderColor='rgba(255,184,48,.35)';badgeBg='rgba(255,184,48,.1)';badgeColor='#FFB830';
    orbNum.style.color='#FFB830';
  } else {
    levelText='High emissions';
    bgGrad='radial-gradient(circle at 40% 38%,rgba(255,107,74,.3),rgba(255,184,48,.1),transparent)';
    borderColor='rgba(255,107,74,.35)';badgeBg='rgba(255,107,74,.1)';badgeColor='#FF6B4A';
    orbNum.style.color='#FF6B4A';
  }
  core.style.background=bgGrad;
  core.style.borderColor=borderColor;
  badge.textContent=levelText;
  badge.style.background=badgeBg;
  badge.style.color=badgeColor;

  const setBar=(fillId,valId,val)=>{
    document.getElementById(fillId).style.width=Math.min(100,(val/8000)*100)+'%';
    document.getElementById(valId).textContent=fmtK(val);
  };
  setBar('gf-transport','gv-transport',e.transport);
  setBar('gf-home',     'gv-home',     e.home);
  setBar('gf-diet',     'gv-diet',     e.diet);
  setBar('gf-lifestyle','gv-lifestyle',e.lifestyle);
}

// attach listeners to all form controls (only exist on predict.html)
const formControls = document.querySelectorAll('select,input[type=number]');
if (formControls.length) {
  formControls.forEach(el=>{
    el.addEventListener('change',updateOrb);
    el.addEventListener('input',updateOrb);
  });
  updateOrb();
}

// Models page — real per-model accuracy, fetched from results_table.csv
// via the backend (see ml_service.get_model_metrics()). Previously this
// grid showed hardcoded placeholder R² values baked into the HTML; this
// replaces them with the real trained-model numbers.
const modelsGrid = document.querySelector('.models-grid');
if (modelsGrid) {
  (async () => {
    const note = document.getElementById('models-metrics-note');
    try {
      const res = await fetch(`${API_BASE_URL}/api/models`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      if (!data.available) {
        modelsGrid.querySelectorAll('.m-card').forEach(card => {
          card.querySelector('.m-card-score').textContent = 'N/A';
          card.querySelector('.m-card-score-label').textContent = 'results_table.csv not found';
        });
        if (note) note.textContent =
          'Real accuracy figures need results_table.csv in backend/models/ — see backend/models/README.md.';
        return;
      }

      let usedCV = false;
      modelsGrid.querySelectorAll('.m-card').forEach(card => {
        const key = card.dataset.model;
        const m = data.models[key];
        if (!m) {
          card.querySelector('.m-card-score').textContent = 'N/A';
          card.querySelector('.m-card-score-label').textContent = 'Not found in results_table.csv';
          return;
        }
        usedCV = usedCV || m.source === 'cv';
        const pct = Math.max(0, Math.min(100, Math.round(m.r2 * 100)));
        card.querySelector('.m-card-score').textContent = `R² ${m.r2.toFixed(2)}`;
        card.querySelector('.m-card-score-label').textContent =
          m.source === 'cv' ? '5-Fold CV Accuracy' : 'Test-Split Accuracy';
        card.querySelector('.m-card-bar-fill').style.width = `${pct}%`;
      });
      if (note) note.textContent = usedCV
        ? 'Figures are 5-fold cross-validated R² on training data — leakage-safe, not a single test-set score.'
        : 'Figures are single test-split R² (CV_R2_mean not present in results_table.csv).';
    } catch (err) {
      modelsGrid.querySelectorAll('.m-card').forEach(card => {
        card.querySelector('.m-card-score').textContent = 'N/A';
        card.querySelector('.m-card-score-label').textContent = 'Backend unreachable';
      });
      if (note) note.textContent =
        `Couldn't reach the backend at ${API_BASE_URL} — make sure Flask is running. See backend/README.md.`;
    }
  })();
}
