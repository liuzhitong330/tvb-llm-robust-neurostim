/* Progressive enhancement: every result and figure remains readable without JS. */
(() => {
  'use strict';
  const chart = document.getElementById('cohort-chart');
  const controls = document.getElementById('cohort-controls');
  const readout = document.getElementById('patient-readout');
  const filter = document.getElementById('cohort-filter');
  const media = window.matchMedia('(max-width: 600px)');
  const colors = {better: '#386e69', worse: '#a35135', muted: '#69636a', grid: '#d8d2d7'};
  let patients = [];
  let order = 'response';
  let subset = 'all';
  const signed = (n, digits = 4) => `${n >= 0 ? '+' : '−'}${Math.abs(n).toFixed(digits)}`;

  function describe(row) {
    return `Patient ${String(row.id).padStart(2, '0')} · ${row.subtype} onset · reward ${row.baseline.toFixed(4)} → ${row.stimulated.toFixed(4)} · change ${signed(row.delta)} (${row.delta > 0 ? 'lower' : 'higher'} variance).`;
  }

  function render() {
    const rows = patients.filter(row => subset === 'all' || row.subtype === subset)
      .sort((a, b) => order === 'patient' ? a.id - b.id : b.delta - a.delta);
    const small = media.matches;
    const width = small ? 430 : 800;
    const height = rows.length * 28 + 115;
    const left = small ? 66 : 147;
    const right = width - (small ? 72 : 110);
    const center = (left + right) / 2;
    const x = value => center + value / .12 * (right - left) / 2;
    const bottom = rows.length * 28 + 40;
    const font = small ? 15 : 14;
    let body = `<title id="cohort-title">Paired virtual-patient responses</title><desc id="cohort-desc">Positive reward change means lower simulated activity variance. These are archived simulation outputs, not clinical outcomes.</desc><g font-family="Arial, Helvetica, sans-serif" font-size="${font}">`;
    const ticks = small ? [-.1, 0, .1] : [-.1, -.05, 0, .05, .1];
    for (const tick of ticks) {
      body += `<line x1="${x(tick)}" y1="40" x2="${x(tick)}" y2="${bottom}" stroke="${tick === 0 ? '#292629' : colors.grid}"/><text x="${x(tick)}" y="${bottom + 25}" fill="${colors.muted}" text-anchor="middle">${tick === 0 ? '0' : signed(tick, 2)}</text>`;
    }
    body += `<text x="${left}" y="22" fill="${colors.worse}">Worse</text><text x="${right}" y="22" text-anchor="end" fill="${colors.better}">Better</text>`;
    for (const [i, row] of rows.entries()) {
      const y = 60 + i * 28;
      const color = row.delta > 0 ? colors.better : colors.worse;
      body += `<g tabindex="0" role="img" data-patient="${row.id}" aria-label="${describe(row)}"><title>${describe(row)}</title><rect class="patient-hit" x="0" y="${y - 13}" width="${width}" height="27" fill="transparent"/><text x="${small ? 4 : 28}" y="${y + 5}" fill="${colors.muted}">P${String(row.id).padStart(2, '0')}</text><line x1="${center}" y1="${y}" x2="${x(row.delta)}" y2="${y}" stroke="${color}" stroke-width="3"/><circle cx="${x(row.delta)}" cy="${y}" r="4.5" fill="${color}"/><text x="${width - 4}" y="${y + 5}" fill="${color}" text-anchor="end">${signed(row.delta)}</text></g>`;
    }
    body += `<text x="${width / 2}" y="${bottom + 57}" fill="${colors.muted}" text-anchor="middle">Change in reward under stimulation</text></g>`;
    // Values come only from the checked-in, numeric display artifact; no user HTML.
    chart.innerHTML = `<svg viewBox="0 0 ${width} ${height}" role="group" aria-labelledby="cohort-title cohort-desc">${body}</svg>`;
    const improved = rows.filter(row => row.delta > 0).length;
    readout.textContent = `${improved} of ${rows.length} improved; ${rows.length - improved} worsened.${subset === 'temporal' ? ' This subgroup contains only two virtual patients.' : ' Select or focus a patient for the paired values.'}`;
  }

  if (chart && controls && readout && filter) {
    fetch('assets/essay/cohort-data.json').then(response => {
      if (!response.ok) throw new Error('Cohort data unavailable');
      return response.json();
    }).then(data => {
      const allowed = ['hippocampal', 'frontal', 'temporal', 'occipital'];
      if (!Array.isArray(data) || data.length !== 20 || !data.every(row =>
        Number.isInteger(row.id) && allowed.includes(row.subtype) &&
        [row.baseline, row.stimulated, row.delta].every(Number.isFinite))) throw new Error('Invalid cohort data');
      patients = data;
      render();
      controls.hidden = false;
      controls.querySelectorAll('[data-order]').forEach(button => {
        button.addEventListener('click', () => {
          order = button.dataset.order;
          controls.querySelectorAll('[data-order]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
          render();
        });
      });
      filter.addEventListener('change', () => { subset = filter.value; render(); });
      media.addEventListener('change', render);
      ['pointerover', 'focusin', 'click'].forEach(event => chart.addEventListener(event, e => {
        const target = e.target.closest('[data-patient]');
        const row = target && patients.find(patient => patient.id === Number(target.dataset.patient));
        if (row) readout.textContent = describe(row);
      }));
    }).catch(() => {
      // Keep the server-rendered figure and its complete data link on failure.
      controls.hidden = true;
    });
  }

  const copy = document.getElementById('copy-citation');
  if (copy && navigator.clipboard && window.isSecureContext) {
    copy.hidden = false;
    copy.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(document.getElementById('bibtex').textContent);
        document.getElementById('copy-status').textContent = ' Copied.';
      } catch {
        document.getElementById('copy-status').textContent = ' Select the citation above to copy it.';
      }
    });
  }

  if ('IntersectionObserver' in window) {
    const links = [...document.querySelectorAll('.contents a[href^="#"]')];
    const observer = new IntersectionObserver(entries => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        links.forEach(link => {
          if (link.hash === `#${entry.target.id}`) link.setAttribute('aria-current', 'location');
          else link.removeAttribute('aria-current');
        });
      }
    }, {rootMargin: '-10% 0px -65% 0px'});
    links.forEach(link => { const element = document.querySelector(link.hash); if (element) observer.observe(element); });
  }
})();
