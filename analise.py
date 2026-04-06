"""
Análise de Evasão e Conclusão — Sistemas de Informação EACH-USP (2005–2025)
Universidade de São Paulo — Escola de Artes, Ciências e Humanidades
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
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, kruskal
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ── Configurações visuais ────────────────────────────────────────────────────
PALETTE = {
    'FUVEST':           '#1f77b4',
    'ENEM_SISU':        '#ff7f0e',
    'TRANSF_INTERNA':   '#2ca02c',
    'TRANSF_EXTERNA':   '#d62728',
    'CONV_PEC_G':       '#9467bd',
    'OUTROS':           '#8c564b',
}

DESFECHO_PALETTE = {
    'Conclusão':              '#2ca02c',
    'Evasão':                 '#d62728',
    'Continua na USP':        '#1f77b4',
    'Saída para outra IES':   '#ff7f0e',
    'Outros':                 '#9467bd',
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
print(f"Colunas: {df.columns.tolist()}")

# Datas já vêm como datetime via openpyxl
df['DATA_INGRESSO'] = pd.to_datetime(df['DATA_INGRESSO'], errors='coerce')
df['DATA_ENCERRAMENTO'] = pd.to_datetime(df['DATA_ENCERRAMENTO'], errors='coerce')
df['ANO_INGRESSO'] = df['DATA_INGRESSO'].dt.year
df['TEMPO_SEMESTRES'] = df['TEMPO_CURSO'] / 180
df['CR_ACUMULADO'] = pd.to_numeric(df['CR_ACUMULADO'], errors='coerce')

# Classificação de desfecho
EVASAO       = {'ABANDONO_FREQUENCIA','ABANDONO_MATRICULA','CANC_CREDITO',
                'CANC_VENCIMENTO','DESISTENCIA','TRANCAMENTO'}
CONCLUSAO    = {'CONCLUSAO'}
CONTINUA_USP = {'REINGRESSO','TRANSF_INT'}
SAIDA_EXT    = {'TRANSF_EXT','CANC_OUTRA_IES'}
OUTROS       = {'FALECIMENTO','DESCUMPRIMENTO_PEC_G'}

def classificar_desfecho(te):
    if te in CONCLUSAO:    return 'Conclusão'
    if te in EVASAO:       return 'Evasão'
    if te in CONTINUA_USP: return 'Continua na USP'
    if te in SAIDA_EXT:    return 'Saída para outra IES'
    if te in OUTROS:       return 'Outros'
    return 'Desconhecido'

df['DESFECHO'] = df['TIPO_ENCERRAMENTO'].apply(classificar_desfecho)

# Flag binária para análise de sobrevivência (evento = evasão)
df['EVADIU'] = df['DESFECHO'].apply(lambda x: 1 if x == 'Evasão' else 0)
df['CONCLUIU'] = df['DESFECHO'].apply(lambda x: 1 if x == 'Conclusão' else 0)

# Coortes recentes (2022+) têm poucos registros — marcar
df['COORTE_RECENTE'] = df['ANO_INGRESSO'] >= 2022

# Grupos com n suficiente para análise inferencial
GRUPOS_PRINCIPAIS = ['FUVEST', 'ENEM_SISU']

print(f"\nDesfechos:\n{df['DESFECHO'].value_counts()}")
print(f"\nIngressos:\n{df['TIPO_INGRESSO'].value_counts()}")
print(f"\nCoortes recentes (2022+): {df['COORTE_RECENTE'].sum()} registros")


# ── 2. INDICADORES DESCRITIVOS ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. INDICADORES DESCRITIVOS")
print("=" * 60)

# 2.1 Taxas gerais
total = len(df)
for d, n in df['DESFECHO'].value_counts().items():
    print(f"  {d}: {n} ({n/total*100:.1f}%)")

# 2.2 Cross-tab ingresso × desfecho
ct = pd.crosstab(df['TIPO_INGRESSO'], df['DESFECHO'])
ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
print("\nCross-tab (%):\n", ct_pct.round(1))

# 2.3 Tempo médio por ingresso e desfecho
tempo_stats = df.groupby(['TIPO_INGRESSO', 'DESFECHO'])['TEMPO_SEMESTRES'].agg(
    ['mean', 'median', 'std', 'count']).round(2)
print("\nTempo (semestres) por ingresso × desfecho:\n", tempo_stats)

# 2.4 CR acumulado por desfecho
cr_stats = df.groupby('DESFECHO')['CR_ACUMULADO'].agg(['mean','median','std','count']).round(2)
print("\nCR acumulado por desfecho:\n", cr_stats)

# Salvar tabelas
ct_pct.round(1).to_csv('tabela_crosstab_pct.csv')
tempo_stats.to_csv('tabela_tempo_stats.csv')
cr_stats.to_csv('tabela_cr_stats.csv')


# ── 3. GRÁFICOS DESCRITIVOS ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. GRÁFICOS DESCRITIVOS")
print("=" * 60)

desfechos_ordem = ['Conclusão','Evasão','Continua na USP','Saída para outra IES','Outros']
ingressos_ordem = df['TIPO_INGRESSO'].value_counts().index.tolist()

# ── Fig 1: Barras empilhadas — desfecho por tipo de ingresso
fig, ax = plt.subplots(figsize=(11, 6))
ct_plot = ct_pct.reindex(columns=[d for d in desfechos_ordem if d in ct_pct.columns],
                          fill_value=0)
ct_plot = ct_plot.reindex(ingressos_ordem)
bottom = np.zeros(len(ct_plot))
for d in ct_plot.columns:
    vals = ct_plot[d].values
    bars = ax.bar(ct_plot.index, vals, bottom=bottom,
                  color=DESFECHO_PALETTE.get(d, '#aaa'), label=d, width=0.6)
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 4:
            ax.text(i, b + v/2, f'{v:.0f}%', ha='center', va='center',
                    fontsize=8.5, color='white', fontweight='bold')
    bottom += vals
ax.set_ylabel('Proporção (%)')
ax.set_xlabel('Forma de Ingresso')
ax.set_title('Distribuição de Desfechos por Forma de Ingresso')
ax.set_ylim(0, 105)
ax.legend(loc='upper right', bbox_to_anchor=(1.18, 1), frameon=True)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f'{OUT}/fig1_desfecho_por_ingresso.png')
plt.close()
print("  Fig 1 salva.")

# ── Fig 2: Distribuição dos motivos de evasão
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

# ── Fig 3: Boxplot de tempo de curso por ingresso e desfecho (principais)
df_box = df[df['TIPO_INGRESSO'].isin(GRUPOS_PRINCIPAIS) &
            df['DESFECHO'].isin(['Conclusão', 'Evasão'])].copy()
fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(data=df_box, x='TIPO_INGRESSO', y='TEMPO_SEMESTRES',
            hue='DESFECHO', palette={'Conclusão': '#2ca02c', 'Evasão': '#d62728'},
            width=0.5, flierprops=dict(marker='o', markersize=3), ax=ax)
ax.set_xlabel('Forma de Ingresso')
ax.set_ylabel('Tempo de Curso (semestres)')
ax.set_title('Distribuição do Tempo de Curso por Ingresso e Desfecho')
ax.legend(title='Desfecho')
plt.tight_layout()
plt.savefig(f'{OUT}/fig3_boxplot_tempo.png')
plt.close()
print("  Fig 3 salva.")

# ── Fig 4: Histograma de CR acumulado por desfecho
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
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

# ── Fig 5: Evolução temporal — taxa de evasão por ano de ingresso (até 2021)
df_coorte = df[df['ANO_INGRESSO'] <= 2021].copy()
evo = df_coorte.groupby('ANO_INGRESSO').apply(
    lambda g: pd.Series({
        'total': len(g),
        'evasao': (g['DESFECHO'] == 'Evasão').sum(),
        'conclusao': (g['DESFECHO'] == 'Conclusão').sum(),
    })
).reset_index()
evo['tx_evasao']   = evo['evasao']   / evo['total'] * 100
evo['tx_conclusao'] = evo['conclusao'] / evo['total'] * 100

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(evo['ANO_INGRESSO'], evo['tx_evasao'],   marker='o', color='#d62728',
        linewidth=2, label='Taxa de Evasão')
ax.plot(evo['ANO_INGRESSO'], evo['tx_conclusao'], marker='s', color='#2ca02c',
        linewidth=2, label='Taxa de Conclusão')
ax.fill_between(evo['ANO_INGRESSO'], evo['tx_evasao'],   alpha=0.1, color='#d62728')
ax.fill_between(evo['ANO_INGRESSO'], evo['tx_conclusao'], alpha=0.1, color='#2ca02c')
ax.set_xlabel('Ano de Ingresso')
ax.set_ylabel('Taxa (%)')
ax.set_title('Evolução das Taxas de Evasão e Conclusão por Coorte de Ingresso (2005–2021)')
ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.xticks(rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUT}/fig5_evolucao_temporal.png')
plt.close()
print("  Fig 5 salva.")

# ── Fig 6: Heatmap de coorte — desfecho por ano × tipo ingresso (FUVEST + ENEM_SISU)
df_heat = df[df['TIPO_INGRESSO'].isin(GRUPOS_PRINCIPAIS) &
             (df['ANO_INGRESSO'] <= 2021)].copy()
pivot = df_heat.pivot_table(index='ANO_INGRESSO', columns='TIPO_INGRESSO',
                             values='EVADIU', aggfunc='mean') * 100
fig, ax = plt.subplots(figsize=(7, 9))
sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', linewidths=0.5,
            cbar_kws={'label': 'Taxa de Evasão (%)'}, ax=ax)
ax.set_title('Taxa de Evasão (%) por Coorte e Forma de Ingresso')
ax.set_xlabel('Forma de Ingresso')
ax.set_ylabel('Ano de Ingresso')
plt.tight_layout()
plt.savefig(f'{OUT}/fig6_heatmap_coorte.png')
plt.close()
print("  Fig 6 salva.")


# ── 4. ANÁLISE DE SOBREVIVÊNCIA (KAPLAN-MEIER) ───────────────────────────────
print("\n" + "=" * 60)
print("4. ANÁLISE DE SOBREVIVÊNCIA — KAPLAN-MEIER")
print("=" * 60)

# Para KM: evento = evasão; censurado = qualquer outro desfecho
# Usamos FUVEST e ENEM_SISU
df_km = df[df['TIPO_INGRESSO'].isin(GRUPOS_PRINCIPAIS)].copy()
df_km = df_km[df_km['TEMPO_SEMESTRES'].notna() & (df_km['TEMPO_SEMESTRES'] > 0)]

fig, ax = plt.subplots(figsize=(10, 6))
kmfs = {}
for grupo, cor in zip(GRUPOS_PRINCIPAIS, ['#1f77b4', '#ff7f0e']):
    sub = df_km[df_km['TIPO_INGRESSO'] == grupo]
    kmf = KaplanMeierFitter()
    kmf.fit(sub['TEMPO_SEMESTRES'], event_observed=sub['EVADIU'], label=grupo)
    kmf.plot_survival_function(ax=ax, color=cor, ci_show=True, linewidth=2)
    kmfs[grupo] = (sub['TEMPO_SEMESTRES'], sub['EVADIU'])
    med = kmf.median_survival_time_
    print(f"  {grupo}: mediana de sobrevivência = {med:.1f} semestres")

ax.set_xlabel('Tempo de Curso (semestres)')
ax.set_ylabel('Probabilidade de Não-Evasão')
ax.set_title('Curvas de Kaplan-Meier — Sobrevivência no Curso\n(evento: evasão)')
ax.set_xlim(0)
ax.set_ylim(0, 1.05)
ax.legend(title='Forma de Ingresso')
plt.tight_layout()
plt.savefig(f'{OUT}/fig7_kaplan_meier.png')
plt.close()
print("  Fig 7 (Kaplan-Meier) salva.")

# Log-rank test
T1, E1 = kmfs['FUVEST']
T2, E2 = kmfs['ENEM_SISU']
lr = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
print(f"\n  Log-rank test FUVEST vs ENEM_SISU:")
print(f"    p-valor = {lr.p_value:.4f} (stat = {lr.test_statistic:.4f})")


# ── 5. TESTES ESTATÍSTICOS INFERENCIAIS ──────────────────────────────────────
print("\n" + "=" * 60)
print("5. TESTES ESTATÍSTICOS INFERENCIAIS")
print("=" * 60)

df_inf = df[df['TIPO_INGRESSO'].isin(GRUPOS_PRINCIPAIS)].copy()

# 5.1 Qui-quadrado: taxa de evasão (evasão vs. não-evasão) entre FUVEST e ENEM_SISU
ct2 = pd.crosstab(df_inf['TIPO_INGRESSO'],
                  df_inf['DESFECHO'].apply(lambda x: 'Evasão' if x == 'Evasão' else 'Não-Evasão'))
chi2, p, dof, expected = chi2_contingency(ct2)
print(f"\n  Qui-quadrado (evasão vs não-evasão):")
print(f"    χ² = {chi2:.4f}, p = {p:.4f}, gl = {dof}")
print(f"    Tabela:\n{ct2}")

# 5.2 Qui-quadrado: conclusão vs. não-conclusão
ct3 = pd.crosstab(df_inf['TIPO_INGRESSO'],
                  df_inf['DESFECHO'].apply(lambda x: 'Conclusão' if x == 'Conclusão' else 'Não-Conclusão'))
chi2_c, p_c, dof_c, _ = chi2_contingency(ct3)
print(f"\n  Qui-quadrado (conclusão vs não-conclusão):")
print(f"    χ² = {chi2_c:.4f}, p = {p_c:.4f}, gl = {dof_c}")

# 5.3 Mann-Whitney: tempo de curso (conclusão) — FUVEST vs ENEM_SISU
t_fuvest_conc = df_inf[(df_inf['TIPO_INGRESSO']=='FUVEST') & (df_inf['DESFECHO']=='Conclusão')]['TEMPO_SEMESTRES'].dropna()
t_enem_conc   = df_inf[(df_inf['TIPO_INGRESSO']=='ENEM_SISU') & (df_inf['DESFECHO']=='Conclusão')]['TEMPO_SEMESTRES'].dropna()
u_stat, p_mw = mannwhitneyu(t_fuvest_conc, t_enem_conc, alternative='two-sided')
print(f"\n  Mann-Whitney (tempo de conclusão — FUVEST vs ENEM_SISU):")
print(f"    U = {u_stat:.1f}, p = {p_mw:.4f}")
print(f"    Mediana FUVEST: {t_fuvest_conc.median():.2f} sem | ENEM_SISU: {t_enem_conc.median():.2f} sem")

# 5.4 Mann-Whitney: tempo de curso (evasão) — FUVEST vs ENEM_SISU
t_fuvest_ev = df_inf[(df_inf['TIPO_INGRESSO']=='FUVEST') & (df_inf['DESFECHO']=='Evasão')]['TEMPO_SEMESTRES'].dropna()
t_enem_ev   = df_inf[(df_inf['TIPO_INGRESSO']=='ENEM_SISU') & (df_inf['DESFECHO']=='Evasão')]['TEMPO_SEMESTRES'].dropna()
u_ev, p_ev = mannwhitneyu(t_fuvest_ev, t_enem_ev, alternative='two-sided')
print(f"\n  Mann-Whitney (tempo até evasão — FUVEST vs ENEM_SISU):")
print(f"    U = {u_ev:.1f}, p = {p_ev:.4f}")
print(f"    Mediana FUVEST: {t_fuvest_ev.median():.2f} sem | ENEM_SISU: {t_enem_ev.median():.2f} sem")

# 5.5 Mann-Whitney: CR acumulado — Conclusão vs Evasão
cr_conc = df[df['DESFECHO']=='Conclusão']['CR_ACUMULADO'].dropna()
cr_ev   = df[df['DESFECHO']=='Evasão']['CR_ACUMULADO'].dropna()
u_cr, p_cr = mannwhitneyu(cr_conc, cr_ev, alternative='two-sided')
print(f"\n  Mann-Whitney (CR acumulado — Conclusão vs Evasão):")
print(f"    U = {u_cr:.1f}, p = {p_cr:.6f}")
print(f"    Mediana Conclusão: {cr_conc.median():.0f} | Evasão: {cr_ev.median():.0f}")


# ── 6. REGRESSÃO LOGÍSTICA ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. REGRESSÃO LOGÍSTICA — PREDIÇÃO DE EVASÃO")
print("=" * 60)

df_reg = df[df['TIPO_INGRESSO'].isin(GRUPOS_PRINCIPAIS) &
            df['DESFECHO'].isin(['Conclusão', 'Evasão'])].copy()
df_reg = df_reg[df_reg['CR_ACUMULADO'].notna() & df_reg['TEMPO_SEMESTRES'].notna()]

# Encode ingresso
df_reg['ING_BIN'] = (df_reg['TIPO_INGRESSO'] == 'ENEM_SISU').astype(int)

X = df_reg[['ING_BIN', 'TEMPO_SEMESTRES', 'CR_ACUMULADO']].values
y = df_reg['EVADIU'].values

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lr_model = LogisticRegression(max_iter=500)
lr_model.fit(X_scaled, y)
scores = cross_val_score(lr_model, X_scaled, y, cv=5, scoring='roc_auc')
print(f"  ROC-AUC (5-fold CV): {scores.mean():.3f} ± {scores.std():.3f}")

coef_names = ['Ingresso (ENEM=1)', 'Tempo no curso', 'Créditos acumulados']
for name, coef in zip(coef_names, lr_model.coef_[0]):
    print(f"  {name}: coef = {coef:.4f}")


# ── 7. ANÁLISE DE COORTE DETALHADA ───────────────────────────────────────────
print("\n" + "=" * 60)
print("7. ANÁLISE DE COORTE DETALHADA (2005–2021)")
print("=" * 60)

df_c = df[df['ANO_INGRESSO'] <= 2021].copy()
coorte_detail = df_c.groupby(['ANO_INGRESSO', 'TIPO_INGRESSO']).apply(
    lambda g: pd.Series({
        'n': len(g),
        'tx_evasao': (g['DESFECHO']=='Evasão').mean()*100,
        'tx_conclusao': (g['DESFECHO']=='Conclusão').mean()*100,
        'tempo_medio_conc': g[g['DESFECHO']=='Conclusão']['TEMPO_SEMESTRES'].mean(),
    })
).reset_index()

print(coorte_detail[coorte_detail['TIPO_INGRESSO'].isin(GRUPOS_PRINCIPAIS)].to_string(index=False))

# ── Fig 8: Evolução por ingresso (FUVEST vs ENEM_SISU)
df_evo2 = coorte_detail[coorte_detail['TIPO_INGRESSO'].isin(GRUPOS_PRINCIPAIS)]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, metric, title, cor_map in zip(
        axes,
        ['tx_evasao', 'tx_conclusao'],
        ['Taxa de Evasão (%)', 'Taxa de Conclusão (%)'],
        [{'FUVEST': '#1f77b4', 'ENEM_SISU': '#ff7f0e'},
         {'FUVEST': '#1f77b4', 'ENEM_SISU': '#ff7f0e'}]):
    for grupo in GRUPOS_PRINCIPAIS:
        sub = df_evo2[df_evo2['TIPO_INGRESSO'] == grupo]
        ax.plot(sub['ANO_INGRESSO'], sub[metric], marker='o',
                color=cor_map[grupo], label=grupo, linewidth=2)
    ax.set_xlabel('Ano de Ingresso')
    ax.set_ylabel(title)
    ax.set_title(f'{title} por Coorte')
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.legend(title='Ingresso')
fig.suptitle('Evolução por Coorte — FUVEST vs ENEM-SISU (2005–2021)', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/fig8_evolucao_fuvest_enem.png')
plt.close()
print("  Fig 8 salva.")

# ── Fig 9: Violin — tempo de conclusão
df_vio = df[df['TIPO_INGRESSO'].isin(GRUPOS_PRINCIPAIS) & (df['DESFECHO']=='Conclusão')]
fig, ax = plt.subplots(figsize=(8, 5))
sns.violinplot(data=df_vio, x='TIPO_INGRESSO', y='TEMPO_SEMESTRES',
               palette={'FUVEST': '#1f77b4', 'ENEM_SISU': '#ff7f0e'},
               inner='quartile', ax=ax)
ax.set_xlabel('Forma de Ingresso')
ax.set_ylabel('Tempo de Conclusão (semestres)')
ax.set_title('Distribuição do Tempo de Conclusão — FUVEST vs ENEM-SISU')
plt.tight_layout()
plt.savefig(f'{OUT}/fig9_violin_conclusao.png')
plt.close()
print("  Fig 9 salva.")

# ── Fig 10: Evasão precoce vs tardia (< 4 semestres vs >= 4)
df_ev = df[df['DESFECHO'] == 'Evasão'].copy()
df_ev['EVASAO_TIPO'] = df_ev['TEMPO_SEMESTRES'].apply(
    lambda x: 'Precoce (< 4 sem)' if x < 4 else 'Tardia (≥ 4 sem)')
ev_split = pd.crosstab(df_ev['TIPO_INGRESSO'], df_ev['EVASAO_TIPO'])
ev_split_pct = ev_split.div(ev_split.sum(axis=1), axis=0) * 100
keep = [i for i in ['FUVEST','ENEM_SISU','TRANSF_INTERNA','TRANSF_EXTERNA'] if i in ev_split_pct.index]
ev_split_pct = ev_split_pct.reindex(keep)

fig, ax = plt.subplots(figsize=(8, 5))
ev_split_pct.plot(kind='bar', ax=ax,
                  color=['#fdae61', '#d73027'], edgecolor='white', width=0.6)
ax.set_xlabel('Forma de Ingresso')
ax.set_ylabel('% dentro do grupo (evadidos)')
ax.set_title('Evasão Precoce vs Tardia por Forma de Ingresso')
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

print("\n" + "=" * 60)
print("ANÁLISE CONCLUÍDA — figuras salvas em ./figuras/")
print("=" * 60)
