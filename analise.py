"""
Análise de Evasão e Conclusão — Sistemas de Informação EACH-USP (2005–2025)
Universidade de São Paulo — Escola de Artes, Ciências e Humanidades
Professor Orientador: Marcelo Morandini
Autor: João Fontanezi
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

PALETTE_ING = {
    'FUVEST':           '#1f77b4',
    'ENEM_SISU':        '#ff7f0e',
    'TRANSF_INTERNA':   '#2ca02c',
    'TRANSF_EXTERNA':   '#d62728',
    'Outros':           '#9467bd',
}

DESFECHO_PALETTE = {
    'Conclusão':             '#2ca02c',
    'Evasão':                '#d62728',
    'Continua na USP':       '#1f77b4',
    'Saída para outra IES':  '#ff7f0e',
    'Outros':                '#9467bd',
}

sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams.update({'figure.dpi': 150, 'savefig.bbox': 'tight'})

OUT = 'figuras'
os.makedirs(OUT, exist_ok=True)

# ── 1. CARGA E PRÉ-PROCESSAMENTO ─────────────────────────────────────────────
print("=" * 60)
print("1. CARGA E PRÉ-PROCESSAMENTO")
print("=" * 60)

df = pd.read_excel('BASE_ALUNOS_FINAL.xlsx')
print(f"Registros carregados: {len(df)}")

df['DATA_INGRESSO']     = pd.to_datetime(df['DATA_INGRESSO'],     errors='coerce')
df['DATA_ENCERRAMENTO'] = pd.to_datetime(df['DATA_ENCERRAMENTO'], errors='coerce')
df['ANO_INGRESSO']      = df['DATA_INGRESSO'].dt.year
df['TEMPO_SEMESTRES']   = df['TEMPO_CURSO'] / 180
df['CR_ACUMULADO']      = pd.to_numeric(df['CR_ACUMULADO'], errors='coerce')

# ── Agrupamento de formas de ingresso ─────────────────────────────────────────

OUTROS_INGRESSO = {'CONV_PEC_G', 'GRADUADO', 'ENEM_USP', 'OLIMP'}

def agrupar_ingresso(ti):
    if ti in OUTROS_INGRESSO:
        return 'Outros'
    return ti

df['INGRESSO'] = df['TIPO_INGRESSO'].apply(agrupar_ingresso)

INGRESSO_ORDEM = ['FUVEST', 'ENEM_SISU', 'TRANSF_INTERNA', 'TRANSF_EXTERNA', 'Outros']

# ── Classificação de desfecho ─────────────────────────────────────────────────
EVASAO       = {'ABANDONO_FREQUENCIA','ABANDONO_MATRICULA','CANC_CREDITO',
                'CANC_VENCIMENTO','DESISTENCIA','TRANCAMENTO'}
CONCLUSAO    = {'CONCLUSAO'}
CONTINUA_USP = {'REINGRESSO','TRANSF_INT'}
SAIDA_EXT    = {'TRANSF_EXT','CANC_OUTRA_IES'}
OUTROS_DES   = {'FALECIMENTO','DESCUMPRIMENTO_PEC_G'}

def classificar_desfecho(te):
    if te in CONCLUSAO:    return 'Conclusão'
    if te in EVASAO:       return 'Evasão'
    if te in CONTINUA_USP: return 'Continua na USP'
    if te in SAIDA_EXT:    return 'Saída para outra IES'
    return 'Outros'

df['DESFECHO'] = df['TIPO_ENCERRAMENTO'].apply(classificar_desfecho)
df['EVADIU']   = (df['DESFECHO'] == 'Evasão').astype(int)
df['CONCLUIU'] = (df['DESFECHO'] == 'Conclusão').astype(int)

GRUPOS_INFER = ['FUVEST', 'ENEM_SISU']

df['COORTE_VALIDA'] = df['ANO_INGRESSO'] <= 2021

print(f"\nDistribuição INGRESSO (agrupado):")
print(df['INGRESSO'].value_counts())
print(f"\nDesfechos:\n{df['DESFECHO'].value_counts()}")


# ── 2. INDICADORES DESCRITIVOS ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. INDICADORES DESCRITIVOS")
print("=" * 60)

total = len(df)
print("\nDesfechos gerais:")
for d, n in df['DESFECHO'].value_counts().items():
    print(f"  {d}: {n} ({n/total*100:.1f}%)")

ct = pd.crosstab(df['INGRESSO'], df['DESFECHO'])
ct = ct.reindex(INGRESSO_ORDEM)
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
print("\nCross-tab % (ingresso agrupado × desfecho):\n", ct_pct.round(1))

tempo_stats = df.groupby(['INGRESSO', 'DESFECHO'])['TEMPO_SEMESTRES'].agg(
    ['mean','median','std','count']).round(2)
print("\nTempo (semestres):\n", tempo_stats)

cr_stats = df.groupby('DESFECHO')['CR_ACUMULADO'].agg(
    ['mean','median','std','count']).round(2)
print("\nCR acumulado:\n", cr_stats)

ct_pct.round(1).to_csv('tabela_crosstab_pct.csv')
tempo_stats.to_csv('tabela_tempo_stats.csv')
cr_stats.to_csv('tabela_cr_stats.csv')


# ── 3. GRÁFICOS DESCRITIVOS ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. GRÁFICOS DESCRITIVOS")
print("=" * 60)

desfechos_ordem = ['Conclusão','Evasão','Continua na USP','Saída para outra IES','Outros']

ct_plot = ct_pct.reindex(columns=[d for d in desfechos_ordem if d in ct_pct.columns], fill_value=0)
fig, ax = plt.subplots(figsize=(11, 6))
bottom = np.zeros(len(ct_plot))
for d in ct_plot.columns:
    vals = ct_plot[d].values
    ax.bar(ct_plot.index, vals, bottom=bottom,
           color=DESFECHO_PALETTE.get(d, '#aaa'), label=d, width=0.6)
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 4:
            ax.text(i, b + v/2, f'{v:.0f}%', ha='center', va='center',
                    fontsize=8.5, color='white', fontweight='bold')
    bottom += vals

for i, ing in enumerate(ct_plot.index):
    n = ct.loc[ing].sum() if ing in ct.index else 0
    ax.text(i, 102, f'n={n}', ha='center', va='bottom', fontsize=8, color='gray')

ax.set_ylabel('Proporção (%)')
ax.set_xlabel('Forma de Ingresso')
ax.set_title('Distribuição de Desfechos por Forma de Ingresso')
ax.set_ylim(0, 110)
ax.legend(loc='upper right', bbox_to_anchor=(1.18, 1), frameon=True)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f'{OUT}/fig1_desfecho_por_ingresso.png')
plt.close()
print("  Fig 1 salva.")

evadidos = df[df['DESFECHO'] == 'Evasão']
motivos = evadidos['TIPO_ENCERRAMENTO'].value_counts()
MOTIVO_LABELS = {
    'CANC_CREDITO':        'Cancelamento\npor créditos',
    'TRANCAMENTO':         'Trancamento\n(sem reingresso)',
    'ABANDONO_MATRICULA':  'Abandono de\nmatrícula',
    'CANC_VENCIMENTO':     'Vencimento\ndo prazo (7 anos)',
    'DESISTENCIA':         'Desistência\nformal',
    'ABANDONO_FREQUENCIA': 'Abandono\npor frequência',
}
fig, ax = plt.subplots(figsize=(10, 5))
labels = [MOTIVO_LABELS.get(m, m) for m in motivos.index]
colors = sns.color_palette('Reds_r', len(motivos))
bars = ax.barh(labels, motivos.values, color=colors)
for bar, v in zip(bars, motivos.values):
    ax.text(bar.get_width() + 4, bar.get_y() + bar.get_height()/2,
            f'{v} ({v/len(evadidos)*100:.1f}%)', va='center', fontsize=9)
ax.set_xlabel('Número de alunos')
ax.set_title('Motivos de Evasão — Sistemas de Informação EACH-USP')
ax.set_xlim(0, motivos.max() * 1.3)
plt.tight_layout()
plt.savefig(f'{OUT}/fig2_motivos_evasao.png')
plt.close()
print("  Fig 2 salva.")

df_box = df[df['INGRESSO'].isin(INGRESSO_ORDEM) &
            df['DESFECHO'].isin(['Conclusão', 'Evasão'])].copy()
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df_box, x='INGRESSO', y='TEMPO_SEMESTRES',
            hue='DESFECHO',
            order=[g for g in INGRESSO_ORDEM if g in df_box['INGRESSO'].unique()],
            palette={'Conclusão': '#2ca02c', 'Evasão': '#d62728'},
            width=0.5, flierprops=dict(marker='o', markersize=3), ax=ax)
ax.set_xlabel('Forma de Ingresso')
ax.set_ylabel('Tempo de Curso (semestres)')
ax.set_title('Distribuição do Tempo de Curso por Ingresso e Desfecho\n'
             '(testes inferenciais aplicados apenas a FUVEST e ENEM_SISU)')
ax.legend(title='Desfecho')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f'{OUT}/fig3_boxplot_tempo.png')
plt.close()
print("  Fig 3 salva.")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, des, cor in zip(axes, ['Conclusão', 'Evasão'], ['#2ca02c', '#d62728']):
    sub = df[df['DESFECHO'] == des]['CR_ACUMULADO'].dropna()
    ax.hist(sub, bins=30, color=cor, alpha=0.8, edgecolor='white')
    ax.axvline(sub.median(), color='black', linestyle='--', linewidth=1.5,
               label=f'Mediana: {sub.median():.0f}')
    ax.set_title(f'CR Acumulado — {des} (n={len(sub)})')
    ax.set_xlabel('Créditos Acumulados')
    ax.set_ylabel('Frequência')
    ax.legend()
plt.suptitle('Distribuição de Créditos Acumulados por Desfecho', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/fig4_hist_cr.png')
plt.close()
print("  Fig 4 salva.")

df_coorte = df[df['COORTE_VALIDA']].copy()
evo = df_coorte.groupby('ANO_INGRESSO').apply(
    lambda g: pd.Series({
        'total':     len(g),
        'evasao':    (g['DESFECHO'] == 'Evasão').sum(),
        'conclusao': (g['DESFECHO'] == 'Conclusão').sum(),
    })
).reset_index()
evo['tx_evasao']    = evo['evasao']    / evo['total'] * 100
evo['tx_conclusao'] = evo['conclusao'] / evo['total'] * 100

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(evo['ANO_INGRESSO'], evo['tx_evasao'],    marker='o', color='#d62728',
        linewidth=2, label='Taxa de Evasão')
ax.plot(evo['ANO_INGRESSO'], evo['tx_conclusao'], marker='s', color='#2ca02c',
        linewidth=2, label='Taxa de Conclusão')
ax.fill_between(evo['ANO_INGRESSO'], evo['tx_evasao'],    alpha=0.1, color='#d62728')
ax.fill_between(evo['ANO_INGRESSO'], evo['tx_conclusao'], alpha=0.1, color='#2ca02c')
ax.set_xlabel('Ano de Ingresso')
ax.set_ylabel('Taxa (%)')
ax.set_title('Evolução das Taxas de Evasão e Conclusão por Coorte (2005–2021)')
ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.xticks(rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/fig5_evolucao_temporal.png')
plt.close()
print("  Fig 5 salva.")

df_heat = df[df['INGRESSO'].isin(GRUPOS_INFER) & df['COORTE_VALIDA']].copy()
pivot = df_heat.pivot_table(index='ANO_INGRESSO', columns='INGRESSO',
                             values='EVADIU', aggfunc='mean') * 100
fig, ax = plt.subplots(figsize=(7, 9))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', linewidths=0.5,
            cbar_kws={'label': 'Taxa de Evasão (%)'}, ax=ax)
ax.set_title('Taxa de Evasão (%) por Coorte e Forma de Ingresso\n(FUVEST e ENEM_SISU)')
ax.set_xlabel('Forma de Ingresso')
ax.set_ylabel('Ano de Ingresso')
plt.tight_layout()
plt.savefig(f'{OUT}/fig6_heatmap_coorte.png')
plt.close()
print("  Fig 6 salva.")


print("\n" + "=" * 60)
print("4. ANÁLISE DE SOBREVIVÊNCIA — KAPLAN-MEIER")
print("=" * 60)

df_km = df[df['INGRESSO'].isin(GRUPOS_INFER)].copy()
df_km = df_km[df_km['TEMPO_SEMESTRES'].notna() & (df_km['TEMPO_SEMESTRES'] > 0)]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ax = axes[0]
kmfs = {}
for grupo, cor in zip(GRUPOS_INFER, ['#1f77b4', '#ff7f0e']):
    sub = df_km[df_km['INGRESSO'] == grupo]
    kmf = KaplanMeierFitter()
    kmf.fit(sub['TEMPO_SEMESTRES'], event_observed=sub['EVADIU'], label=grupo)
    kmf.plot_survival_function(ax=ax, color=cor, ci_show=True, linewidth=2)
    kmfs[grupo] = (sub['TEMPO_SEMESTRES'], sub['EVADIU'])
    print(f"  {grupo}: mediana sobrevivência = {kmf.median_survival_time_:.1f} sem")
ax.set_xlabel('Tempo de Curso (semestres)')
ax.set_ylabel('Probabilidade de Não-Evasão')
ax.set_title('KM — FUVEST vs ENEM_SISU\n(inferencial)')
ax.set_xlim(0); ax.set_ylim(0, 1.05)
ax.legend(title='Ingresso')

ax2 = axes[1]
cores_km = {'FUVEST':'#1f77b4','ENEM_SISU':'#ff7f0e',
            'TRANSF_INTERNA':'#2ca02c','TRANSF_EXTERNA':'#d62728','Outros':'#9467bd'}
df_km_all = df[df['TEMPO_SEMESTRES'].notna() & (df['TEMPO_SEMESTRES'] > 0)].copy()
for grupo in INGRESSO_ORDEM:
    sub = df_km_all[df_km_all['INGRESSO'] == grupo]
    if len(sub) < 5:
        continue
    kmf = KaplanMeierFitter()
    kmf.fit(sub['TEMPO_SEMESTRES'], event_observed=sub['EVADIU'],
            label=f'{grupo} (n={len(sub)})')
    kmf.plot_survival_function(ax=ax2, color=cores_km[grupo], ci_show=False,
                               linewidth=2 if grupo in GRUPOS_INFER else 1.2,
                               linestyle='-' if grupo in GRUPOS_INFER else '--')
ax2.set_xlabel('Tempo de Curso (semestres)')
ax2.set_ylabel('Probabilidade de Não-Evasão')
ax2.set_title('KM — Todos os grupos\n(descritivo)')
ax2.set_xlim(0); ax2.set_ylim(0, 1.05)
ax2.legend(title='Ingresso', fontsize=8)

fig.suptitle('Curvas de Kaplan-Meier — Sobrevivência no Curso (evento: evasão)', y=1.01)
plt.tight_layout()
plt.savefig(f'{OUT}/fig7_kaplan_meier.png')
plt.close()
print("  Fig 7 (Kaplan-Meier) salva.")

T1, E1 = kmfs['FUVEST']
T2, E2 = kmfs['ENEM_SISU']
lr = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
print(f"\n  Log-rank FUVEST vs ENEM_SISU: p={lr.p_value:.4f} (stat={lr.test_statistic:.4f})")


# ── 5. TESTES INFERENCIAIS ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. TESTES ESTATÍSTICOS INFERENCIAIS (FUVEST vs ENEM_SISU)")
print("=" * 60)

df_inf = df[df['INGRESSO'].isin(GRUPOS_INFER)].copy()

ct2 = pd.crosstab(df_inf['INGRESSO'],
                  df_inf['DESFECHO'].apply(lambda x: 'Evasão' if x=='Evasão' else 'Não-Evasão'))
chi2, p, dof, _ = chi2_contingency(ct2)
print(f"\n  χ² evasão:    χ²={chi2:.4f}, p={p:.4f}, gl={dof}")
print(f"  Tabela:\n{ct2}")

ct3 = pd.crosstab(df_inf['INGRESSO'],
                  df_inf['DESFECHO'].apply(lambda x: 'Conclusão' if x=='Conclusão' else 'Não-Conclusão'))
chi2_c, p_c, _, _ = chi2_contingency(ct3)
print(f"\n  χ² conclusão: χ²={chi2_c:.4f}, p={p_c:.4f}")

t_fuv_c = df_inf[(df_inf['INGRESSO']=='FUVEST')    & (df_inf['DESFECHO']=='Conclusão')]['TEMPO_SEMESTRES'].dropna()
t_ene_c = df_inf[(df_inf['INGRESSO']=='ENEM_SISU') & (df_inf['DESFECHO']=='Conclusão')]['TEMPO_SEMESTRES'].dropna()
u1, p1 = mannwhitneyu(t_fuv_c, t_ene_c, alternative='two-sided')
print(f"\n  Mann-Whitney tempo conclusão: U={u1:.0f}, p={p1:.4f}")
print(f"    Mediana FUVEST={t_fuv_c.median():.2f} | ENEM_SISU={t_ene_c.median():.2f} sem")

t_fuv_e = df_inf[(df_inf['INGRESSO']=='FUVEST')    & (df_inf['DESFECHO']=='Evasão')]['TEMPO_SEMESTRES'].dropna()
t_ene_e = df_inf[(df_inf['INGRESSO']=='ENEM_SISU') & (df_inf['DESFECHO']=='Evasão')]['TEMPO_SEMESTRES'].dropna()
u2, p2 = mannwhitneyu(t_fuv_e, t_ene_e, alternative='two-sided')
print(f"\n  Mann-Whitney tempo evasão:    U={u2:.0f}, p={p2:.4f}")
print(f"    Mediana FUVEST={t_fuv_e.median():.2f} | ENEM_SISU={t_ene_e.median():.2f} sem")

cr_c = df[df['DESFECHO']=='Conclusão']['CR_ACUMULADO'].dropna()
cr_e = df[df['DESFECHO']=='Evasão']['CR_ACUMULADO'].dropna()
u3, p3 = mannwhitneyu(cr_c, cr_e, alternative='two-sided')
print(f"\n  Mann-Whitney CR (conclusão vs evasão): U={u3:.0f}, p={p3:.6f}")
print(f"    Mediana Conclusão={cr_c.median():.0f} | Evasão={cr_e.median():.0f}")


# ── 6. REGRESSÃO LOGÍSTICA ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. REGRESSÃO LOGÍSTICA — PREDIÇÃO DE EVASÃO")
print("=" * 60)

df_reg = df[df['INGRESSO'].isin(GRUPOS_INFER) &
            df['DESFECHO'].isin(['Conclusão','Evasão']) &
            df['CR_ACUMULADO'].notna() &
            df['TEMPO_SEMESTRES'].notna()].copy()

df_reg['ING_BIN'] = (df_reg['INGRESSO'] == 'ENEM_SISU').astype(int)
X = df_reg[['ING_BIN','TEMPO_SEMESTRES','CR_ACUMULADO']].values
y = df_reg['EVADIU'].values

scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X)
modelo  = LogisticRegression(max_iter=500)
scores  = cross_val_score(modelo, X_sc, y, cv=5, scoring='roc_auc')
modelo.fit(X_sc, y)

print(f"  ROC-AUC (5-fold CV): {scores.mean():.3f} ± {scores.std():.3f}")
for name, coef in zip(['Ingresso (ENEM=1)','Tempo no curso','Créditos acumulados'],
                      modelo.coef_[0]):
    print(f"  {name}: coef = {coef:.4f}")


# ── 7. ANÁLISE DE COORTE DETALHADA (2005–2021) ───────────────────────────────
print("\n" + "=" * 60)
print("7. ANÁLISE DE COORTE DETALHADA (2005–2021)")
print("=" * 60)

df_c = df[df['COORTE_VALIDA']].copy()
coorte_det = df_c.groupby(['ANO_INGRESSO','INGRESSO']).apply(
    lambda g: pd.Series({
        'n':            len(g),
        'tx_evasao':    (g['DESFECHO']=='Evasão').mean()*100,
        'tx_conclusao': (g['DESFECHO']=='Conclusão').mean()*100,
        'tempo_conc':   g[g['DESFECHO']=='Conclusão']['TEMPO_SEMESTRES'].mean(),
    })
).reset_index()

print(coorte_det[coorte_det['INGRESSO'].isin(GRUPOS_INFER)].to_string(index=False))

df_evo2 = coorte_det[coorte_det['INGRESSO'].isin(GRUPOS_INFER)]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, metric, ylabel in zip(axes,
        ['tx_evasao','tx_conclusao'],
        ['Taxa de Evasão (%)','Taxa de Conclusão (%)']):
    for grupo in GRUPOS_INFER:
        sub = df_evo2[df_evo2['INGRESSO']==grupo]
        ax.plot(sub['ANO_INGRESSO'], sub[metric], marker='o',
                color=PALETTE_ING[grupo], label=grupo, linewidth=2)
    ax.set_xlabel('Ano de Ingresso')
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel + ' por Coorte')
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.legend(title='Ingresso')
fig.suptitle('Evolução por Coorte — FUVEST vs ENEM-SISU (2005–2021)', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/fig8_evolucao_fuvest_enem.png')
plt.close()
print("  Fig 8 salva.")

df_vio = df[df['INGRESSO'].isin(GRUPOS_INFER) & (df['DESFECHO']=='Conclusão')]
fig, ax = plt.subplots(figsize=(8, 5))
sns.violinplot(data=df_vio, x='INGRESSO', y='TEMPO_SEMESTRES',
               palette={g: PALETTE_ING[g] for g in GRUPOS_INFER},
               inner='quartile', ax=ax)
ax.set_xlabel('Forma de Ingresso')
ax.set_ylabel('Tempo de Conclusão (semestres)')
ax.set_title('Distribuição do Tempo de Conclusão — FUVEST vs ENEM-SISU')
plt.tight_layout()
plt.savefig(f'{OUT}/fig9_violin_conclusao.png')
plt.close()
print("  Fig 9 salva.")

df_ev = df[df['DESFECHO']=='Evasão'].copy()
df_ev['EVASAO_TIPO'] = df_ev['TEMPO_SEMESTRES'].apply(
    lambda x: 'Precoce (< 4 sem)' if x < 4 else 'Tardia (≥ 4 sem)')

grupos_ev = ['FUVEST','ENEM_SISU','TRANSF_INTERNA','TRANSF_EXTERNA']
ev_split = pd.crosstab(df_ev[df_ev['INGRESSO'].isin(grupos_ev)]['INGRESSO'],
                       df_ev[df_ev['INGRESSO'].isin(grupos_ev)]['EVASAO_TIPO'])
ev_split_pct = ev_split.div(ev_split.sum(axis=1), axis=0) * 100
keep = [g for g in grupos_ev if g in ev_split_pct.index]
ev_split_pct = ev_split_pct.reindex(keep)

fig, ax = plt.subplots(figsize=(9, 5))
ev_split_pct.plot(kind='bar', ax=ax,
                  color=['#fdae61','#d73027'], edgecolor='white', width=0.6)
ax.set_xlabel('Forma de Ingresso')
ax.set_ylabel('% dentro do grupo (evadidos)')
ax.set_title('Evasão Precoce vs Tardia por Forma de Ingresso\n'
             '(análise descritiva — TRANSF com n pequeno)')
ax.legend(title='Tipo de Evasão')
plt.xticks(rotation=15)
for p in ax.patches:
    if p.get_height() > 2:
        ax.annotate(f'{p.get_height():.0f}%',
                    (p.get_x() + p.get_width()/2., p.get_height()),
                    ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUT}/fig10_evasao_precoce_tardia.png')
plt.close()
print("  Fig 10 salva.")

ev2 = df[df['DESFECHO']=='Evasão'][df[df['DESFECHO']=='Evasão']['INGRESSO'].isin(GRUPOS_INFER)].copy()
motivo_ing = pd.crosstab(ev2['TIPO_ENCERRAMENTO'], ev2['INGRESSO'])
motivo_ing_pct = motivo_ing.div(motivo_ing.sum(axis=0), axis=1) * 100
motivo_ing_pct.index = [MOTIVO_LABELS.get(m, m) for m in motivo_ing_pct.index]

fig, ax = plt.subplots(figsize=(10, 6))
motivo_ing_pct.plot(kind='barh', ax=ax,
                    color=[PALETTE_ING['FUVEST'], PALETTE_ING['ENEM_SISU']],
                    edgecolor='white', width=0.6)
ax.set_xlabel('% dentro do grupo (evadidos)')
ax.set_title('Distribuição dos Motivos de Evasão — FUVEST vs ENEM_SISU')
ax.legend(title='Ingresso')
plt.tight_layout()
plt.savefig(f'{OUT}/fig11_motivos_evasao_por_ingresso.png')
plt.close()
print("  Fig 11 salva.")

print("\n" + "=" * 60)
print("ANÁLISE CONCLUÍDA — figuras salvas em ./figuras/")
print("=" * 60)
