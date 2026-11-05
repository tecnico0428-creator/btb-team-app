// 한국어 글자 읽기(PaddleOCR PP-OCRv5 한국어 인식 + 글자 영역 찾기 PP-OCRv3 det, onnxruntime-web). 모두 이 기기 안에서 실행됩니다.
(function () {
  const P = { ready: false };
  function loadScript(src) { return new Promise((res, rej) => { const s = document.createElement('script'); s.src = src; s.onload = res; s.onerror = () => rej(new Error('분석 도구를 불러오지 못했습니다: ' + src)); document.head.appendChild(s); }); }
  async function fetchBuf(url, cb) {
    const r = await fetch(url); if (!r.ok) throw new Error('파일을 내려받지 못했습니다(' + r.status + ')');
    const total = +r.headers.get('content-length') || 0;
    if (!r.body || !r.body.getReader) return await r.arrayBuffer();
    const rd = r.body.getReader(), chunks = []; let got = 0;
    for (;;) { const { done, value } = await rd.read(); if (done) break; chunks.push(value); got += value.length; if (cb) cb(got, total); }
    const out = new Uint8Array(got); let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; } return out.buffer;
  }
  P.init = async function (cfg, prog) {
    if (P.ready) return;
    prog = prog || function () {};
    if (!window.ort) { prog(0.02, '분석 도구를 불러오는 중...'); await loadScript(cfg.ort); }
    ort.env.wasm.wasmPaths = cfg.wasmBase; ort.env.wasm.numThreads = 1; ort.env.wasm.proxy = false;
    const sizes = { det: 2.4, rec: 13.4 }; let done = 0;
    const part = (name, w) => (g, t) => prog(0.05 + 0.4 * (done + (t ? g / t : 0.5) * w) / (sizes.det + sizes.rec), '인식 모델을 내려받는 중... (처음 한 번)');
    const detB = await fetchBuf(cfg.det, part('det', sizes.det)); done += sizes.det;
    const recB = await fetchBuf(cfg.rec, part('rec', sizes.rec)); done += sizes.rec;
    const dictT = await (await fetch(cfg.dict)).text();
    prog(0.5, '모델을 준비하는 중...');
    const opt = { executionProviders: ['wasm'], graphOptimizationLevel: 'all' };
    P.det = await ort.InferenceSession.create(detB, opt);
    P.rec = await ort.InferenceSession.create(recB, opt);
    const lines = dictT.split('\n').map(s => s.replace(/\r$/, '')); if (lines.length && lines[lines.length - 1] === '') lines.pop();
    P.dict = ['blank'].concat(lines); P.dict.push(' ');
    P.ready = true;
  };
  function toCanvas(src, w, h) { const c = document.createElement('canvas'); c.width = w; c.height = h; const g = c.getContext('2d', { willReadFrequently: true }); g.imageSmoothingQuality = 'high'; g.fillStyle = '#fff'; g.fillRect(0, 0, w, h); g.drawImage(src, 0, 0, w, h); return c; }
  function detTensor(cv) {
    const w = cv.width, h = cv.height, d = cv.getContext('2d').getImageData(0, 0, w, h).data, n = w * h, f = new Float32Array(3 * n);
    const mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
    for (let i = 0, p = 0; i < d.length; i += 4, p++) { f[p] = (d[i + 2] / 255 - mean[0]) / std[0]; f[n + p] = (d[i + 1] / 255 - mean[1]) / std[1]; f[2 * n + p] = (d[i] / 255 - mean[2]) / std[2]; }
    return new ort.Tensor('float32', f, [1, 3, h, w]);
  }
  function detBoxes(prob, w, h) {
    const thr = 0.3, bin = new Uint8Array(w * h); for (let i = 0; i < bin.length; i++) bin[i] = prob[i] > thr ? 1 : 0;
    const lab = new Int32Array(w * h), boxes = [], stack = new Int32Array(w * h); let id = 0;
    for (let y0 = 0; y0 < h; y0++) for (let x0 = 0; x0 < w; x0++) {
      const s = y0 * w + x0; if (!bin[s] || lab[s]) continue; id++; let sp = 0; stack[sp++] = s; lab[s] = id;
      let minX = x0, maxX = x0, minY = y0, maxY = y0, area = 0, sum = 0;
      while (sp) { const q = stack[--sp]; const x = q % w, y = (q - x) / w; area++; sum += prob[q]; if (x < minX) minX = x; if (x > maxX) maxX = x; if (y < minY) minY = y; if (y > maxY) maxY = y;
        if (x > 0 && bin[q - 1] && !lab[q - 1]) { lab[q - 1] = id; stack[sp++] = q - 1; } if (x < w - 1 && bin[q + 1] && !lab[q + 1]) { lab[q + 1] = id; stack[sp++] = q + 1; }
        if (y > 0 && bin[q - w] && !lab[q - w]) { lab[q - w] = id; stack[sp++] = q - w; } if (y < h - 1 && bin[q + w] && !lab[q + w]) { lab[q + w] = id; stack[sp++] = q + w; } }
      const bw = maxX - minX + 1, bh = maxY - minY + 1; if (area < 12 || Math.min(bw, bh) < 3 || sum / area < 0.55) continue;
      const dist = (bw * bh * 1.6) / (2 * (bw + bh)); boxes.push([minX - dist, minY - dist, maxX + 1 + dist, maxY + 1 + dist]);
    }
    return boxes;
  }
  async function recBox(src, box) {
    const x0 = Math.max(0, Math.floor(box[0])), y0 = Math.max(0, Math.floor(box[1])), x1 = Math.min(src.width, Math.ceil(box[2])), y1 = Math.min(src.height, Math.ceil(box[3]));
    const w = x1 - x0, h = y1 - y0; if (w < 4 || h < 4) return { text: '', conf: 0 };
    const H = 48, W = Math.max(16, Math.min(2400, Math.ceil(H * w / h)));
    const c = document.createElement('canvas'); c.width = W; c.height = H; const g = c.getContext('2d', { willReadFrequently: true }); g.imageSmoothingQuality = 'high'; g.fillStyle = '#fff'; g.fillRect(0, 0, W, H); g.drawImage(src, x0, y0, w, h, 0, 0, W, H);
    const d = g.getImageData(0, 0, W, H).data, n = W * H, f = new Float32Array(3 * n);
    for (let i = 0, p = 0; i < d.length; i += 4, p++) { f[p] = (d[i + 2] / 255 - 0.5) / 0.5; f[n + p] = (d[i + 1] / 255 - 0.5) / 0.5; f[2 * n + p] = (d[i] / 255 - 0.5) / 0.5; }
    const out = await P.rec.run({ [P.rec.inputNames[0]]: new ort.Tensor('float32', f, [1, 3, H, W]) });
    const o = out[P.rec.outputNames[0]]; const T = o.dims[1], C = o.dims[2], a = o.data; let last = 0, text = '', conf = 0, cnt = 0;
    for (let t = 0; t < T; t++) { let bi = 0, bv = -1e9; const off = t * C; for (let k = 0; k < C; k++) { const v = a[off + k]; if (v > bv) { bv = v; bi = k; } } if (bi !== 0 && bi !== last) { text += P.dict[bi] || ''; conf += bv; cnt++; } last = bi; }
    return { text, conf: cnt ? conf / cnt : 0 };
  }
  // 이미지(캔버스/이미지)를 읽어 글자 줄 목록으로 돌려준다. 같은 높이의 글자 조각은 한 줄로 묶는다.
  P.recognize = async function (img, prog) {
    prog = prog || function () {};
    const W0 = img.naturalWidth || img.width, H0 = img.naturalHeight || img.height, long = Math.max(W0, H0);
    const target = long < 1000 ? 1280 : Math.min(long, 1536); const k = target / long;
    const dw = Math.max(32, Math.round(W0 * k / 32) * 32), dh = Math.max(32, Math.round(H0 * k / 32) * 32);
    const dc = toCanvas(img, dw, dh);
    prog(0.55, '글자 위치를 찾는 중...');
    const out = await P.det.run({ [P.det.inputNames[0]]: detTensor(dc) });
    const pm = out[P.det.outputNames[0]]; const boxesSmall = detBoxes(pm.data, dw, dh);
    const sx = W0 / dw, sy = H0 / dh; const src = (img instanceof HTMLCanvasElement) ? img : toCanvas(img, W0, H0);
    const boxes = boxesSmall.map(b => [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy]).sort((a, b) => (a[1] + a[3]) - (b[1] + b[3]));
    const items = [];
    for (let i = 0; i < boxes.length; i++) { const r = await recBox(src, boxes[i]); if (r.text.trim()) items.push({ box: boxes[i], text: r.text.trim(), conf: r.conf }); if (i % 4 === 0) { prog(0.6 + 0.38 * i / boxes.length, '글자를 읽는 중... ' + (i + 1) + '/' + boxes.length); await new Promise(r => setTimeout(r, 0)); } }
    // 줄 묶기
    const hs = items.map(it => it.box[3] - it.box[1]).sort((a, b) => a - b), medH = hs.length ? hs[Math.floor(hs.length / 2)] : 20;
    items.sort((a, b) => ((a.box[1] + a.box[3]) / 2) - ((b.box[1] + b.box[3]) / 2));
    const rows = [];
    for (const it of items) { const yc = (it.box[1] + it.box[3]) / 2; const row = rows.find(r => Math.abs(r.yc - yc) < medH * 0.55); if (row) { row.items.push(it); row.yc = (row.yc * (row.items.length - 1) + yc) / row.items.length; } else rows.push({ yc, items: [it] }); }
    rows.sort((a, b) => a.yc - b.yc);
    const lines = rows.map(r => r.items.sort((a, b) => a.box[0] - b.box[0]).map(i => i.text).join('  |  '));
    return { lines, items, boxes: boxes.length };
  };
  window.PPOCR = P;
})();
