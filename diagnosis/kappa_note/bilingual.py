"""Interleave note/info_scale.tex (EN) and note/info_scale_zh.tex (ZH) segment by segment;
writes note/info_scale_bilingual.tex (compile in note/ with xelatex or tectonic)."""
import re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
EN = open(HERE / 'note/info_scale.tex').read()
ZH = open(HERE / 'note/info_scale_zh.tex').read()

def body(t):
    b = t.split('\\vspace{-1.5em}', 1)[1]
    return b.split('\\begin{thebibliography}', 1)[0]

MARK = re.compile(r'^(\\section|\\subsection|\\textbf\{|\\begin\{(table|figure)\})', re.M)

def segments(t):
    t = body(t)
    out, cur = [], []
    floating = False
    for line in t.splitlines():
        if not floating and MARK.match(line) and cur:
            out.append('\n'.join(cur).strip()); cur = []
        if re.match(r'\\begin\{(table|figure)\}', line):
            floating = True
        cur.append(line)
        if re.match(r'\\end\{(table|figure)\}', line):
            floating = False
            out.append('\n'.join(cur).strip()); cur = []
    if cur: out.append('\n'.join(cur).strip())
    return [s for s in out if s]

en, zh = segments(EN), segments(ZH)
print(len(en), len(zh), file=sys.stderr)
for i, (a, b) in enumerate(zip(en, zh)):
    ka, kb = MARK.match(a), MARK.match(b)
    ta = ka.group(1) if ka else 'text'; tb = kb.group(1) if kb else 'text'
    if ta != tb:
        print('MISMATCH', i, a[:60], '|', b[:60], file=sys.stderr); sys.exit(1)

def strip_labels(s):
    s = re.sub(r'\s*\\label\{[^}]*\}', '', s)
    return s.replace('\\begin{equation}', '\\begin{equation*}').replace('\\end{equation}', '\\end{equation*}')

def caption(s):
    m = re.search(r'\\caption\{', s)
    i = m.end(); depth = 1; j = i
    while depth:
        depth += {'{': 1, '}': -1}.get(s[j], 0); j += 1
    return m.start(), j, s[i:j-1]

parts = []
for a, b in zip(en, zh):
    if a.startswith('\\begin{table') or a.startswith('\\begin{figure'):
        s0, s1, ca = caption(a)
        _, _, cb = caption(b)
        parts.append(a[:s0] + '\\caption{' + ca + '\\newline{\\zhfont ' + cb + '}}' + a[s1:])
        continue
    head = ''
    m = re.match(r'\\(sub)?section\{(.*)\}\s*\n?', a)
    if m:
        mb = re.match(r'\\(sub)?section\{(.*)\}\s*\n?', b)
        lab = re.search(r'\\label\{[^}]*\}', a)
        head = '\\%ssection{%s\\quad{\\zhfont %s}}\n' % (m.group(1) or '', m.group(2), mb.group(2))
        if lab: head += lab.group(0) + '\n'
        a = a[m.end():]; b = b[mb.end():]
        a = re.sub(r'^\\label\{[^}]*\}\s*', '', a)
        b = re.sub(r'^\\label\{[^}]*\}\s*', '', b)
    parts.append(head + (a.strip() + '\n\n\\begin{zhbox}\n' + strip_labels(b.strip()) + '\n\\end{zhbox}\n' if a.strip() else ''))

pre = r'''\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,booktabs,graphicx}
\usepackage[section]{placeins}
\usepackage{xeCJK}
\setCJKmainfont{Songti SC}
\setCJKsansfont{PingFang SC}
\usepackage[round]{natbib}
\usepackage[hidelinks]{hyperref}
\usepackage[most]{tcolorbox}
\newtcolorbox{zhbox}{breakable,colback=black!4,colframe=black!4,boxrule=0pt,left=6pt,right=6pt,top=3pt,bottom=3pt,arc=0pt,before skip=2pt,after skip=10pt,fontupper=\linespread{1.2}\selectfont}
\newcommand{\zhfont}{}
\graphicspath{{figures/}}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0.4em}
\setlength{\emergencystretch}{2em}
\raggedbottom
\newcommand{\Rhat}{\hat R}
\title{An information scale for the mixing of the Default BART sampler\\[0.3em]\large 中英对照版（灰底为中文）}
\author{}
\date{5 November 2026}
\begin{document}
\maketitle
\vspace{-1.5em}
'''
bib = '\\begin{thebibliography}' + EN.split('\\begin{thebibliography}', 1)[1]
open(HERE / 'note/info_scale_bilingual.tex', 'w').write(pre + '\n\n'.join(parts) + '\n' + bib)
