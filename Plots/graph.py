from graphviz import Digraph

dot = Digraph(comment='YOLOspine Architecture (Compressed)', format='pdf')
dot.attr(rankdir='LR', size='8.27,3.9', nodesep='0.15', ranksep='0.25', splines='ortho', fontname='Arial', bgcolor='white')
dot.attr(label='\n\nYOLOspine: A Two-Stage Detection Framework for Spinal Pathologies', fontsize='18', fontname='Arial Bold')

# Input
dot.node('Input', 'Input: 384×384×3\n(RGB Image)', shape='box', style='filled,rounded', fillcolor='lightblue', fontname='Arial', fontsize='12', rank='source')

# Backbone: Split into 2 vertical groups (5 nodes each)
with dot.subgraph(name='cluster_backbone') as b:
    b.attr(label='Backbone: Feature Extraction', style='bold,rounded', fontname='Arial Bold', fontsize='14', color='gray60', bgcolor='aliceblue')
    with b.subgraph(name='backbone_g1') as g1:
        g1.attr(rank='same')
        g1.node('B1', 'Conv 3×3, 64\n384×384×64', shape='box', style='filled,rounded', fillcolor='lightcyan', fontsize='12')
        g1.node('B2', 'MaxPool 2×2\n192×192×64', shape='box', style='filled,rounded', fillcolor='lightcyan', fontsize='12')
        g1.node('B3', 'C3 192×192×128\nSiLU, Skip', shape='box', style='filled,rounded', fillcolor='lightcyan', fontsize='12')
        g1.node('B4', 'MaxPool 2×2\n96×96×128', shape='box', style='filled,rounded', fillcolor='lightcyan', fontsize='12')
        g1.node('B5', 'C3 96×96×256\nSiLU, Skip', shape='box', style='filled,rounded', fillcolor='lightcyan', fontsize='12')
        g1.edges([('B1', 'B2'), ('B2', 'B3'), ('B3', 'B4'), ('B4', 'B5')])
    with b.subgraph(name='backbone_g2') as g2:
        g2.attr(rank='same')
        g2.node('B6', 'MaxPool 2×2\n48×48×256 (P3)', shape='box', style='filled,rounded', fillcolor='palegreen', color='forestgreen', penwidth='2', fontsize='12')
        g2.node('B7', 'R-ELAN\n48×48×256', shape='box', style='filled,rounded', fillcolor='lightcyan', fontsize='12')
        g2.node('B8', 'MaxPool 2×2\n24×24×256', shape='box', style='filled,rounded', fillcolor='lightcyan', fontsize='12')
        g2.node('B9', 'R-ELAN\n24×24×512 (P4)', shape='box', style='filled,rounded', fillcolor='palegreen', color='forestgreen', penwidth='2', fontsize='12')
        g2.node('B10', 'MaxPool 2×2\n12×12×512 (P5)', shape='box', style='filled,rounded', fillcolor='palegreen', color='forestgreen', penwidth='2', fontsize='12')
        g2.edges([('B6', 'B7'), ('B7', 'B8'), ('B8', 'B9'), ('B9', 'B10')])
    b.edge('B5', 'B6', constraint='true')

# Neck: Split into 2 vertical groups (5 nodes each)
with dot.subgraph(name='cluster_neck') as n:
    n.attr(label='Neck: Feature Refinement', style='bold,rounded', fontname='Arial Bold', fontsize='14', color='gray60', bgcolor='lavenderblush')
    with n.subgraph(name='neck_g1') as g1:
        g1.attr(rank='same')
        g1.node('N1', 'P5 Conv 3×3\n12×12×512', shape='box', style='filled,rounded', fillcolor='lavender', fontsize='12')
        g1.node('N2', 'Upsample\n24×24×512', shape='box', style='filled,rounded', fillcolor='lavender', fontsize='12')
        g1.node('N3', 'Concat P4\n24×24×1024', shape='box', style='filled,rounded', fillcolor='lavender', fontsize='12')
        g1.node('N4', 'R-ELAN\n24×24×256', shape='box', style='filled,rounded', fillcolor='lavender', fontsize='12')
        g1.node('N5', 'A2 Attention\n24×24×256 (P4\')', shape='box', style='filled,rounded', fillcolor='palegreen', color='forestgreen', penwidth='2', fontsize='12')
        g1.edges([('N1', 'N2'), ('N2', 'N3'), ('N3', 'N4'), ('N4', 'N5')])
    with n.subgraph(name='neck_g2') as g2:
        g2.attr(rank='same')
        g2.node('N6', 'Upsample\n48×48×256', shape='box', style='filled,rounded', fillcolor='lavender', fontsize='12')
        g2.node('N7', 'Concat P3\n48×48×512', shape='box', style='filled,rounded', fillcolor='lavender', fontsize='12')
        g2.node('N8', 'R-ELAN\n48×48×256', shape='box', style='filled,rounded', fillcolor='lavender', fontsize='12')
        g2.node('N9', 'A2 Attention\n48×48×256 (P3\')', shape='box', style='filled,rounded', fillcolor='palegreen', color='forestgreen', penwidth='2', fontsize='12')
        g2.node('N10', 'Downsample\n12×12×512 (P5\')', shape='box', style='filled,rounded', fillcolor='palegreen', color='forestgreen', penwidth='2', fontsize='12')
        g2.edges([('N6', 'N7'), ('N7', 'N8'), ('N8', 'N9')])
    n.edge('N5', 'N6', constraint='true')
    n.edge('N5', 'N10', style='dashed', color='darkblue', penwidth='1.5', constraint='false')

# Stage 1 Head: Single vertical stack
with dot.subgraph(name='cluster_s1') as s1:
    s1.attr(label='Stage 1 Head', style='bold,rounded', fontname='Arial Bold', fontsize='14', color='gray60', bgcolor='honeydew', rankdir='TB')
    s1.node('S1A', "P3' Head\n48×48×(C1+5)", shape='box', style='filled,rounded', fillcolor='lightgreen', fontsize='12')
    s1.node('S1B', "P4' Head\n24×24×(C1+5)", shape='box', style='filled,rounded', fillcolor='lightgreen', fontsize='12')
    s1.node('S1C', "P5' Head\n12×12×(C1+5)", shape='box', style='filled,rounded', fillcolor='lightgreen', fontsize='12')

# Stage 2 Head: Single vertical stack, cascaded from Stage 1
with dot.subgraph(name='cluster_s2') as s2:
    s2.attr(label='Stage 2 Head', style='bold,rounded', fontname='Arial Bold', fontsize='14', color='gray60', bgcolor='lightyellow', rankdir='TB')
    s2.node('S2A', 'RoIAlign\n7×7×256', shape='box', style='filled,rounded', fillcolor='khaki', fontsize='12')
    s2.node('S2B', 'Cls: Sigmoid\nReg: 5\nN2 × (C2+5)', shape='box', style='filled,rounded', fillcolor='khaki', fontsize='12')
    s2.edge('S2A', 'S2B', penwidth='2', color='darkblue')

# Post-processing
dot.node('PP', 'Soft-NMS\nPreserve Pathologies', shape='box', style='filled,rounded', fillcolor='peachpuff', fontname='Arial', fontsize='12', color='sienna', penwidth='2', rank='sink')

# Connections
flow = [
    ('Input', 'B1'), ('B10', 'N1'),
    ('B6', 'N7', 'P3', 'dashed'), ('B9', 'N3', 'P4', 'dashed'),
    ('N9', 'S1A'), ('N5', 'S1B'), ('N10', 'S1C'),
    ('S1A', 'S2A', 'Positive Boxes', 'dashed'), ('S1B', 'S2A', None, 'dashed'), ('S1C', 'S2A', None, 'dashed'),
    ('S2B', 'PP')
]
for edge in flow:
    if len(edge) == 2:
        src, tgt = edge
        dot.edge(src, tgt, color='darkblue', penwidth='2')
    else:
        src, tgt, label, style = edge
        dot.edge(src, tgt, label=label, style=style, color='darkblue', penwidth='2', constraint='false')

# Save the diagram as PDF (default)
dot.render(filename='yolospine_academic_diagram', format='pdf', cleanup=False)

# Optional: Save as EMF for PowerPoint editing (uncomment to use)
# dot.render(filename='yolospine_academic_diagram', format='emf', cleanup=False)