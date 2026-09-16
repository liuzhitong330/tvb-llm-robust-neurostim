"""Build editorial SVG figures from published JSON; never run or alter simulations.

Run from the repository root: python3 scripts/build_essay_figures.py
"""
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets' / 'essay'
INK, MUTED, GRID = '#292629', '#69636a', '#d8d2d7'
PURPLE, RUST, GREEN = '#695575', '#a35135', '#386e69'


def load(name):
    return json.loads((ROOT / 'results' / name).read_text())


def text(x, y, value, size=16, color=INK, anchor='start', extra=''):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}" {extra}>{html.escape(str(value))}</text>'


def line(x1, y1, x2, y2, color=GRID, width=1, extra=''):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" {extra}/>'


def svg(name, width, height, title, desc, body):
    content = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc"><title id="title">{html.escape(title)}</title><desc id="desc">{html.escape(desc)}</desc><g font-family="Arial, Helvetica, sans-serif">' + ''.join(body) + '</g></svg>'
    (OUT / name).write_text(content)


def network():
    data = load('brain3d_data.json')
    body = ['<rect width="1100" height="350" fill="#eae8f0"/>']
    body += [text(46,48,'THE EXPERIMENT',13,PURPLE,extra='letter-spacing="1.8"'),text(46,145,'Propose a setting.',25),text(46,176,'Literature-informed hypotheses',16,MUTED),text(785,145,'Test the weakest result.',24),text(785,176,'Across sampled virtual patients',16,MUTED),text(550,320,'76-region TVB network',14,MUTED,'middle')]
    body += [line(298,160,362,160,'#888092'),text(355,165,'›',21,'#888092'),line(717,160,765,160,'#888092'),text(758,165,'›',21,'#888092')]
    coords = {r['id']:(550+r['y']*1.53,160+r['x']*1.45) for r in data['regions']}
    for e in data['edges']:
        a,b = coords[e['i']],coords[e['j']]
        body.append(line(*a,*b,PURPLE,.65,'opacity=".13"'))
    for r in data['regions']:
        x,y = coords[r['id']]
        body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{PURPLE}"><title>{html.escape(r["label"])}</title></circle>')
    svg('virtual-network.svg',1100,350,'From a proposal to a virtual-brain experiment','Network geometry is drawn from the archived TVB connectivity: 76 nodes and their stored edges.',body)
    mobile = (OUT/'virtual-network.svg').read_text().replace('viewBox="0 0 1100 350"','viewBox="375 0 350 350"')
    (OUT/'virtual-network-mobile.svg').write_text(mobile)


def comparison():
    intrinsic = load('results.json')
    external = load('results_v2.json')
    base = -intrinsic[0]['worst_case_reward']
    best = -max(v['worst_case_reward'] for v in intrinsic)
    ext = -external['best_worst']
    body = []
    x = lambda v: 195 + v/0.6*570
    for tick in [0,.2,.4,.6]:
        body += [line(x(tick),75,x(tick),350),text(x(tick),385,f'{tick:.1f}',14,MUTED,'middle')]
    body += [text(0,27,'A  ·  Intrinsic model control',19),text(0,51,'Change epileptogenicity and coupling',14,MUTED)]
    body += [text(0,230,'B  ·  External-current stimulation',19),text(0,254,'Add a current offset at one region',14,MUTED)]
    for y,label,value,color in [(100,'Baseline',base,'#b3abae'),(150,'Selected setting',best,PURPLE),(296,'No stimulation',base,'#b3abae'),(346,'Right hippocampus',ext,RUST)]:
        body += [text(0,y+5,label,15,MUTED),f'<rect x="195" y="{y-12}" width="{x(value)-195}" height="24" fill="{color}"/>',text(x(value)+12,y+5,f'{value:.4f}',16,color)]
    body += [text(195,199,'39.8% lower worst-case proxy score',16,PURPLE),text(195,417,'1.7% lower worst-case proxy score',16,RUST),text(195,455,'Simulated activity variance · lower is better',14,MUTED)]
    svg('two-experiments.svg',850,470,'Two different kinds of control produce different gains','Intrinsic: 0.5285 to 0.3182, reported 39.8% reduction. External current: 0.5285 to 0.5194, 1.7% reduction. Positive variance scores are the negatives of the archived rewards. The two plots share one scale.',body)


def trajectory():
    rows=load('results.json'); body=[]
    x=lambda i:65+i*91
    y=lambda v:250-(v+.56)/.32*200
    for v in [-.5,-.4,-.3]:
        body += [line(60,y(v),730,y(v)),text(48,y(v)+5,f'{v:.1f}',14,MUTED,'end')]
    for key,color,label in [('mean_reward','#aaa1a8','Mean'),('worst_case_reward',PURPLE,'Worst case')]:
        points=' '.join(f'{x(i)},{y(r[key])}' for i,r in enumerate(rows))
        body += [f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>']
        for i,r in enumerate(rows):body += [f'<circle cx="{x(i)}" cy="{y(r[key])}" r="3.5" fill="{color}"/>']
        body += [text(737,y(rows[-1][key])+5,label,14,color)]
    for i in range(8):body += [text(x(i),280,'Baseline' if i==0 else str(i),13,MUTED,'middle')]
    body += [text(60,28,'Reward · closer to zero is better',15,MUTED),text(400,310,'Baseline, followed by seven proposals',14,MUTED,'middle')]
    svg('intrinsic-trajectory.svg',840,328,'The archived intrinsic search trajectory','Eight evaluations, including baseline. Each candidate is evaluated across five virtual patients. Mean and worst-case rewards are plotted separately.',body)


def cohort():
    data=load('cohort_results_20.json')
    rows=[{'id':i+1,'baseline':a,'stimulated':b,'delta':round(b-a,4),'subtype':data['soz_types'][i]} for i,(a,b) in enumerate(zip(data['baseline']['rewards'],data['optimized']['rewards']))]
    (OUT/'cohort-data.json').write_text(json.dumps(rows,indent=2)+'\n')
    rows.sort(key=lambda r:r['delta'],reverse=True)
    body=[]; x=lambda v:420+v/.12*270
    for tick in [-.10,-.05,0,.05,.10]:
        body += [line(x(tick),42,x(tick),602,INK if tick==0 else GRID,1),text(x(tick),626,f'{tick:+.2f}' if tick else '0',14,MUTED,'middle')]
    body += [text(147,23,'Higher variance',14,RUST),text(690,23,'Lower variance',14,GREEN,'end')]
    for i,row in enumerate(rows):
        y=60+i*28; c=GREEN if row['delta']>0 else RUST
        body += [text(28,y+5,f'P{row["id"]:02}',14,MUTED),line(420,y,x(row['delta']),y,c,3),f'<circle cx="{x(row["delta"])}" cy="{y}" r="4.5" fill="{c}"/>',text(766,y+5,f'{row["delta"]:+.4f}',14,c,'end')]
    body += [text(420,657,'Change in reward under stimulation',14,MUTED,'middle')]
    svg('patient-responses.svg',800,675,'Different virtual patients respond differently','All 20 paired responses. Eleven improve and nine worsen. Positive reward change means lower simulated activity variance. Stored patient IDs refer to original row order.',body)



def comparison_mobile():
    intrinsic = load('results.json')
    external = load('results_v2.json')
    base = -intrinsic[0]['worst_case_reward']
    best = -max(v['worst_case_reward'] for v in intrinsic)
    ext = -external['best_worst']
    x = lambda v: 12 + v/.6*330
    body=[]
    for top,title,gain,val,color in [(0,'A · Intrinsic model control','39.8% lower proxy score',best,PURPLE),(220,'B · External-current stimulation','1.7% lower proxy score',ext,RUST)]:
        body += [text(12,top+24,title,19),text(12,top+49,gain,16,color)]
        for y,label,v,c in [(top+94,'Baseline',base,'#b3abae'),(top+158,'Selected setting',val,color)]:
            body += [text(12,y-16,label,15,MUTED),f'<rect x="12" y="{y-7}" width="{x(v)-12}" height="21" fill="{c}"/>',text(x(v)+10,y+9,f'{v:.4f}',15,c)]
        body += [line(12,top+191,342,top+191)]
        for tick in [0,.2,.4,.6]:body += [text(x(tick),top+210,f'{tick:.1f}',13,MUTED,'middle')]
    body += [text(12,461,'Activity variance · lower is better',15,MUTED)]
    svg('two-experiments-mobile.svg',420,477,'Two different kinds of control','Both comparisons share a zero-to-0.6 axis. Intrinsic: 0.5285 to 0.3182. External current: 0.5285 to 0.5194.',body)

if __name__ == '__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    network(); comparison(); comparison_mobile(); trajectory(); cohort()
    print('Built five SVG figures and the cohort display data from unchanged archived results.')
