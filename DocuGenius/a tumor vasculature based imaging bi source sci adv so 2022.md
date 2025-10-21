## Page 1

SCIENCE ADVANCES | RESEARCH ARTICLE
HEALTH AND MEDICINE
Copyright © 2022
The Authors, some
A tumor vasculature–based imaging biomarker rights reserved;
exclusive licensee
for predicting response and survival in patients American Association
for the Advancement
with lung cancer treated with checkpoint inhibitors of Science. No claim to
original U.S. Government
Mehdi Alilou1*, Mohammadhadi Khorrami2, Prateek Prasanna3, Kaustav Bera1, Works. Distributed
Amit Gupta4, Vidya Sankar Viswanathan2, Pradnya Patil5, Priya Darsini Velu6, Pingfu Fu7, under a Creative
Commons Attribution
Vamsidhar Velcheti8, Anant Madabhushi2,9*
NonCommercial
License 4.0 (CC BY-NC).
Tumor vasculature is a key component of the tumor microenvironment that can influence tumor behavior and
therapeutic resistance. We present a new imaging biomarker, quantitative vessel tortuosity (QVT), and evaluate
its association with response and survival in patients with non–small cell lung cancer (NSCLC) treated with immune
checkpoint inhibitor (ICI) therapies. A total of 507 cases were used to evaluate different aspects of the QVT
biomarkers. QVT features were extracted from computed tomography imaging of patients before and after ICI
therapy to capture the tortuosity, curvature, density, and branching statistics of the nodule vasculature. Our
results showed that QVT features were prognostic of OS (HR = 3.14, 0.95% CI = 1.2 to 9.68, P = 0.0006, C-index = 0.61)
and could predict ICI response with AUCs of 0.66, 0.61, and 0.67 on three validation sets. Our study shows that
QVT imaging biomarker could potentially aid in predicting and monitoring response to ICI in patients with NSCLC.
INTRODUCTION unmet clinical need for biomarkers to identify patients who are most
Immune checkpoint inhibitors (ICIs) have revolutionized the treat- likely to benefit from ICIs and determine the potential nonrespond-
ment paradigm in non–small cell lung cancer (NSCLC) (1, 2) and ers who can be spared both the financial cost and side effects of IO
are now the standard of care either alone or in combination with (8, 9). While exploratory markers such as tumor mutational burden
chemotherapy as the first-line therapy in treatment-naïve patients (10) are now being evaluated, there continues to be a need for iden-
and as the second-line therapy in chemotherapy-refractive patients tifying noninvasive biomarkers, both for predicting and monitoring
(3). As of 2018, almost every patient with advanced NSCLC without response to ICIs.
targetable mutations is treated with ICI either in the first-line set- Radiomics refers to the process of image analysis that results in
ting or as subsequent lines of therapy. First-line platinum-based high-throughput extraction of subvisual and quantitative features
cytotoxic chemotherapy for patients with advanced NSCLC pro- from radiologic scans including x-rays, computed tomography (CT),
duces unstable responses at best (4, 5). ultrasound, and magnetic resonance imaging. Recently, radiomic
Landmark clinical trials leading to the approval of ICIs in NSCLC approaches have been applied in the context of prognosticating out-
have demonstrated an association between Programmed cell death come and predicting response to IO (11–14). Most of these studies
ligand 1 (PD-L1) expression and response to ICI; hence, PD-L1 tu- have analyzed the tumoral shape and textural radiomic features
mor expression levels are used as routine clinical predictive bio- for predicting response and outcome in patients with NSCLC. For
marker for deciding the regimen of anti–PD-L1 immunotherapy example, the authors in (15) presented a method involving changes in
(IO) (6). However, these trials demonstrated that some patients the textural radiomic features of CT images to predict overall survival
with low PD-L1 expression can still have clinical responses with (OS) and response to IO in patients with NSCLC. Trebeschi et al.
these agents. This has now led to adding chemotherapy to ICIs in (11, 16) showed the utility of radiomic features in predicting response
patients with PD-L1 tumor expression less than 50% and ICI mono- to IO and patient outcomes in metastatic NSCLC.
therapy for PD-L1 greater than 50% as the standard of care. How- Tumor vasculature is a key component of the tumor microenvi-
ever, response rates to ICI monotherapy remain modest (27% in ronment that can influence invasiveness, metastatic potential, and
PD-L1–positive NSCLC in the first-line setting, 45% in PD-L1–high therapeutic refractoriness. Cancer cells encourage the growth of blood
subgroup, and 19% in the second-line setting) (7). There is thus an vessels to feed the tumor by producing vascular endothelial growth
factor, thus creating an immune-excluded phenotype of tumors
(17, 18). In the immune-excluded phenotype, there are underlying
mechanical or chemical barriers between infiltrating lymphocytes
1Department of Biomedical Engineering, Case Western Reserve University, Cleveland,
OH 44106, USA. 2Department of Biomedical Engineering, Emory University, and the tumor site in which antiangiogenesis or particular antistro-
Atlanta, GA 30322, USA. 3Department of Biomedical Informatics, Stony Brook Uni- mal therapy might be of benefit in enhancing the efficacy of IO (19).
versity, Stony Brook, NY 11790, USA. 4University Hospitals Cleveland Medical Center, A promising strategy in anticancer therapy is tumor blood vessel
Case Western Reserve University, Cleveland, OH 44106, USA. 5Department of Solid
Tumor Oncology, Cleveland Clinic, Cleveland, OH 44106, USA. 6Pathology and Lab- normalization. Many studies showed how less twisted vessels are
oratory Medicine, Weill Cornell Medicine Physicians, New York, NY 10021, USA. able to counteract metastasis formation and favor chemotherapeutic
7Department of Population and Quantitative Health Sciences, CWRU, Cleveland, drug delivery to tumors (20). In addition, it has been demonstrated
OH 44106, USA. 8Department of Hematology and Oncology, NYU Langone Health, that aberrant vessel morphology potentiates treatment resistance and
New York, NY 10016, USA. 9Atlanta Veterans Administration Medical Center, Atlanta, GA
lack of durable therapeutic response by reducing drug transfer to the
30322, USA.
*Corresponding author. Email: me.alilou@gmail.com (M.A.); anantm@emory.edu (A.M.) tumor bed (21).
Alilou et al., Sci. Adv. 8, eabq4609 (2022) 25 November 2022 1 of 11
Downloaded
from
https://www.science.org
on
November
28,
2022

## Page 2

SCIENCE ADVANCES | RESEARCH ARTICLE
In this work, we present and validate quantitative vessel tortuosity curvature, tortuosity, branching, and distribution of acute and ob-
(QVT), a new imaging biomarker for predicting the response and tuse angles measured from each of the three consecutive points on
outcome prognosis of patients with NSCLC treated with ICI thera- the vessels’ centerline. The QVT risk score (QRS) stratified patients
pies. We hypothesize that the tumor vasculature is more twisted in into high- and low-risk groups in D1 with P = 0.0006, hazard ratio
nonresponders compared to responders to ICI. In addition, we hy- (HR) = 3.14 (95% CI = 1.2 to 9.68), and C-index = 0.61. The QRS
pothesize that the vasculature twistedness on nonresponders to ICI was found to be prognostic in three validation sets (D2, D3, and D4)
causes antitumor T cells to accumulate at the tumor site but fail to with HR = 2.49 (0.95% CI = 1.17 to 5.32), P = 0.002, and C-index =
efficiently infiltrate the tumor accounting for therapeutic refracto- 0.62 (difference of median OS = 10 months) in D2; HR = 2.12 (0.95%
riness. In this study, we sought to evaluate our QVT imaging biomarker CI = 1.04 to 4.29), P = 0.014, and C-index = 0.61 (difference of me-
on a total number of 507 NSCLC cases in terms of predicting response, dian OS = 14 months) in D3; and HR = 2.98 (0.95% CI = 1.12 to
monitoring response, and prognosticating outcome. Toward this 7.93), and P = 0.04 (difference of median OS = 6.3 months) in D4.
end, we used contrast-enhanced CT scans from 162 patients with The multivariable OS analysis results for the QVT features are shown
advanced NSCLC before and after two to three cycles of PD-1/ in Table 2. In addition, a Cox proportional hazards analysis yielded
PD-L1 ICI therapy from three sites. The association of QVT and PD-L1 HR = 2.22 (95% CI = 1.38 to 3.6), P = 0.001, and C-index = 0.7 in
expression (PD-L1low and PD-L1high) was evaluated using a set of predicting OS for patients in D7 who underwent a combination of
204 patients with early-stage NSCLC. We analyzed the QVT associa- ICI and chemotherapy. A multivariable analysis with combination
tion with gene set enrichment analysis (GSEA) pathways on a set of of PD-L1 expression and QRS revealed that QRS is the only variable
92 patients with early-stage NSCLC with available RNA sequencing that is significantly associated with OS in D7 [QRS: HR = 2.17 (95%
data. In addition, we evaluated the prognostic and predictive poten- CI = 1.33 to 3.55), P = 0.001; PD-L1: HR = 1.47 (95% CI = 0.46 to
tial of QVT on a set of 45 patients with NSCLC who underwent a 4.7), P = 0.51, and C-index = 0.71]. Figure 1 (B to E) illustrates a
combination of ICI and chemotherapy. Kaplan-Meier estimation of OS via QRS in low- and high-risk
patients in each of D1, D2, D3, and D7.
RESULTS Experiment 2: Association between delta QVT features,
Experiment 1: Association of baseline QVT features response to ICI, and OS
with response to ICI and OS The ICI response prediction model trained with delta QVT features
A linear discriminant analysis (LDA) machine learning classifier was yielded an AUC of 0.92 (95% CI = 0.84 to 0.97) and 0.85 (95%
used to determine the ability of the selected QVT features in discrimi- CI = 0.71 to 0.99) on D2 and D3, respectively. Figure 1F shows the
nation of patients into responders and nonresponders to ICI. The receiver operating characteristic (ROC) curve of the response pre-
response prediction classifier yielded an area under the curve (AUC) diction models trained with baseline and delta QVT features. The
of 0.74 [95% confidence interval (CI) = 0.73 to 0.75] on the training multivariable response prediction results of delta QVT features are
(D1) and corresponding AUCs of 0.66 (95% CI = 0.58 to 0.81), 0.61 illustrated in Table 1.
(95% CI = 0.56 to 0.78), and 0.67 (95% CI = 0.59 to 0.88) on D2, D3, Looking at the distribution of angles measured from any three
and D4 validation sets, respectively, for patients treated with ICI consecutive points of the vasculature, we observed that in the pre-
monotherapy. The classifier’s performance on predicting response treatment scans of the responders to therapy, the distribution of obtuse
for patients treated with different ICI agents is illustrated in Table 1. angles was almost doubled compared with nonresponders, which means
In addition, using a model trained with QVT features on D1 that responders primarily consisted of less tortuous vessel branches
(ICI monotherapy), the model had an AUC = 0.64 (95% CI = 0.57 to (Fig. 2, A and B). Section SA gives more details on the QVT features.
0.82) in predicting response to chemoimmunotherapy on D7. Analysis of the changes in vessel tortuosity between pre- and post-
Figure 1A shows a feature expression cluster heatmap of the most treatment scans revealed that the number of acute angles associated
discriminating QVT features for responder and nonresponder pa- with the vessels was significantly reduced after treatment in respond-
tients in the training set (D1). As may be observed, a number of ers, while these acute angles remained nearly the same or increased
baseline QVT features showed statistically significant differential after treatment in nonresponders (Fig. 2, C and D). Figure 3 illustrates
expression between responders and nonresponders in D1. three-dimensional (3D) tortuosity and curvature maps for responders
Seven stable baseline QVT features were found to be prognostic and nonresponders between pre- and posttreatment scans. The mean
of OS. These features corresponded to statistics of vasculature curvature and torsion values for the responder case decreased after
Table 1. Multiagent response prediction analysis (with metric AUC) of QVT features on validation sets. AUC is reported for two models including the
baseline QVT model trained with pretreatment scans and the delta QVT model trained with both pre- and posttreatment scans.
AUC on D2 AUC on D3 AUC on D4
Dataset
Baseline Delta Baseline Delta Baseline Delta
All agents 0.66 0.92 0.61 0.85 0.67 –
Nivolumab 0.67 0.90 0.77 0.85 0.62 –
Pembrolizumab 0.90 0.73 0.63 0.72 0.95 –
Atezolizumab – – 0.65 – –
Alilou et al., Sci. Adv. 8, eabq4609 (2022) 25 November 2022 2 of 11
Downloaded
from
https://www.science.org
on
November
28,
2022

## Page 3

SCIENCE ADVANCES | RESEARCH ARTICLE
Fig. 1. Association between delta QVT features, response to ICI, and OS. (A) Unsupervised clustering of QVT features and patients revealed two dominant patient
groups with high- and low-risk groups. High-risk group is associated and aligned with nonresponders to ICI. (B to E) Kaplan-Meier survival curves represent a significant
difference in OS between patients with low and high QRS on D1, D2, D3, and D7 sets. (F) ROC curves of response prediction model trained with QVT (AUCs of 0.66 and 0.61
on D2 and D3, respectively) and delta QVT (AUCs of 0.92 and 0.85 on D2 and D3, respectively).
complete description of features can be found in the Supplementary
Table 2. Multiagent OS analysis of QVT features on validation sets. Materials. Delta QVT features were found to be prognostic of OS on
Dataset D2 (N = 50) D3 (N = 27) D4 (N = 23) D2 and D3 with HR = 2.64 (95% CI = 1.37 to 5.1) and P = 0.008 and
HR = 0.245 (0.0647 to 0.925) and P = 0.0006, respectively.
Nivolumab HR = 2.64 HR = 2.08 HR = 3.11
(1.37–5.1) (1.12–4.45) (1.17–8.28)
Experiment 3: Molecular, histological, and radiogenomic
P = 0.00867 P = 0.021 P = 0.0368
underpinning of QVT
Pembrolizumab HR = 2.08 HR = 2.27 – Three QVT features were found to be strongly associated with
(1.912–4.75) (1.969–5.3)
PD-L1 expression. As shown in Fig. 4 (A to C) , statistically signif-
P = 0.026 P = 0.00987 icant differences were found between QVT features of the PD-L1low
Atezolizumab – – – and PD-L1high groups (P < 0.002). Moreover, significant differ-
ence was found between QRS features of the low– and high–PD-L1
groups (P = 0.0023). In addition, the QVT features were found to
be strongly associated with tumor-infiltrating lymphocyte (TIL)
treatment with respect to pretreatment scan, while the same feature density on hematoxylin and eosin (H&E) images of baseline bi-
increased after treatment for the nonresponder case. In addition, opsy scans.
QVT features f7, f8, f11, and f12 corresponding to statistics of vessel The mean of the TIL grouping factor was found to be statistically
curvature were found to be significantly different between respond- significantly correlated with a QVT_1 feature (r = −0.56 and P = 0.001).
ers and nonresponders in terms of both baseline and delta QVT fea- The same feature also showed moderately high correlation with
tures (fig. S5, A and B). Moreover, the distribution of acute angles QVT (QVT_6) feature. Both QVT features refer to the distribution
(f68) for baseline and curvature statistics (f9 and 10) and distribu- of the acute angles of the vasculature in which QVT_1 and QVT_6
tion of obtuse angles (f52) of delta QVT features were also signifi- refer to the distribution of the acute angles within the range of 1° to
cantly different between responders and nonresponders to ICI. The 12° and 61° to 72°. Figure 4D shows the corresponding correlation
Alilou et al., Sci. Adv. 8, eabq4609 (2022) 25 November 2022 3 of 11
Downloaded
from
https://www.science.org
on
November
28,
2022

## Page 4

SCIENCE ADVANCES | RESEARCH ARTICLE
Fig. 2. Responder tumor to ICI has less tortuous vessel branches compared to nonresponder. (A and B) CT scans of a responder and nonresponder to IO with corre-
sponding 3D rendering of the tumors and the corresponding vasculature. (C and D) The distribution of angles between any three consecutive points of the vasculature
for all responder (C) and nonresponder patients (D), which was measured for both pretreatment (blue bars) and posttreatment (red bars) CT scans. Angles were binned
from 1 to 15 with 1 referring to the most obtuse (wide) angles and 15 referring to the most acute angles.
matrix. The matrix represents TIL and QVT features arranged in activity of T cells. In this work, we present a noninvasive quantitative
rows and columns, respectively. The correlation value between each measurement of vessel twistedness or tortuosity as a novel imaging
pair of TILs and QVT features is presented as circles in the corre- analytic technique to evaluate the association between vessel con-
sponding matrix element. volutedness, response to ICI therapy, and OS of patients with ad-
We found that wingless-integrated (WNT) signaling pathway vanced NSCLC.
was up-regulated in the high-QVT phenotype group. In addition, A few studies have investigated the role of radiomic texture fea-
blood vessel morphogenesis and the fibroblast growth factor (FGF), tures of the nodule on CT scans in predicting response to different
an angiogenic growth factor (22), pathway were found to be up- therapies. Authors in (13) found that the skewness of the intensity
regulated in the high-QVT phenotype group. histogram measured in Hounsfield units and relative volume of air
in the segmented tumor were associated with treatment response.
They also found that response to IO was correlated negatively with
DISCUSSION the tumor convexity and positively with the edge–to–core size ratio.
Recent evidence indicates that angiogenesis, lymphangiogenesis, Trebeschi et al. (11) used radiomic features on contrast-enhanced CT
and the tumor environment have important immunomodulatory scans to predict responses to IO in patients with metastatic NSCLC
roles that contribute to the immune evasion of tumors (17). It has treated in a second-line setting. Their results showed that more het-
been also observed that angiogenesis, invasion, and vessel prolif- erogeneous tumors with irregular patterns of intensities have a bet-
eration might be important regulators of PD-L1 expression, given ter OS. In another study, Tang et al. (12) presented an approach for
the association of these processes with malignant progression (23). developing a predictor of OS in patients with NSCLC based on a
It is well acknowledged that most tumors trigger an immune re- pathology-informed radiomic model. Sun et al. (14) showed a radio-
sponse modulated by TILs. Previous studies have reported an asso- mic approach to assess tumor-infiltrating CD8 cells and response to
ciation between higher density of TILs in patients and favorable anti–PD-1 or anti–PD-L1 IO.
responses to IO (24, 25). It has been shown that the tumor vascu- Our work differs from these previous approaches in that it in-
lature can actively suppress antitumor immune responses (26), and volves using mathematical measurements from nodule vasculature
expression of vascular adhesion molecules in blood vessels correlates (QVT) to predict OS and distinguish responders from nonresponders
with the TIL density in the tumor microenvironment. The structur- in patients with NSCLC treated with ICI, as opposed to radiomic
ally and functionally aberrant tumor vasculature contributes to the texture–based measurements. Because ICIs work by modulating the
protumorigenic and immunosuppressive tumor microenvironment PD-L1 axis, we also investigated the association of QVT features with
by maintaining a cancer cell’s permissive environment character- PD-L1 expression in early-stage ICI-naïve patients and found that
ized by hypoxia, acidosis, and high interstitial pressure while simul- QVT was strongly associated with PD-L1 expression. Our results
taneously generating a physical barrier to T cell infiltration. Recent show that most acute angles of the blood vessel vasculature are also
research has also shown that blood endothelial cells forming the inversely associated with TIL expression. This is in line with favorable
tumor vessels can actively suppress the recruitment, adhesion, and responders to ICI having higher TIL expression and corresponding
Alilou et al., Sci. Adv. 8, eabq4609 (2022) 25 November 2022 4 of 11
Downloaded
from
https://www.science.org
on
November
28,
2022

## Page 5

SCIENCE ADVANCES | RESEARCH ARTICLE
Fig. 3. 3D tortuosity and curvature maps for responder and nonresponder before and after treatment. Baseline and posttreatment scans of a responder and non-
responder to ICI are illustrated in the left. Second column from the left represents the 3D renderings of the tumors and associated vasculature. Third and fourth columns
show the curvature and tortuosity maps of the vasculature that reflects the regional curvature of vessels in 3D and the extent of convolutedness of each vessel, respec-
tively. The mean curvature and torsion values for the responder case decreased after treatment with respect to the pretreatment scan, while the same feature increased
after treatment for the nonresponder case.
lower QVT. Moreover, radiogenomic analysis of the QVT features QVT biomarker was validated in two independent test sets, accrued
showed that tumors with a highly tortuous vasculature are associated from three different sites and across three different IO agents. In
with WNT signaling pathway, blood vessel morphogenesis, and the addition to ICI monotherapy, we showed the potential of QVT to
FGF pathway. Activation of WNT signaling pathway has been shown be predictive of response and prognostic of survival in patients
to correlate with immune exclusion (27) across human cancers, which treated with first-line combination chemotherapy + IO. This study
are usually more aggressive cancers often with refractory to therapy. uniquely provides a cellular, molecular, and genomic underpinning
This study is the first to demonstrate that tumor vessel tortuosity for the imaging-derived QVT features that were identified to be
measurements extracted from routine contrast CT images are asso- associated with response to ICI therapy. We also assessed the sta-
ciated with response to ICIs and prognostic of OS in patients with bility of QVT features in test-retest scans and then measured their
metastatic NSCLC treated with ICI. Our group has also previously stability against segmentation errors. As illustrated in section SB, 22
shown the utility of an initial version of QVT features in distinguish- of 74 QVT features were found to be moderately stable with an intra-
ing benign and malignant nodules in patients with NSCLC (28). The class correlation coefficient of >0.4. The number of stable features
Alilou et al., Sci. Adv. 8, eabq4609 (2022) 25 November 2022 5 of 11
Downloaded
from
https://www.science.org
on
November
28,
2022

## Page 6

SCIENCE ADVANCES | RESEARCH ARTICLE
Fig. 4. Molecular, histological, and radiogenomic underpinning of QVT features. (A to C) Statistically significant differences were found between QVT features of the
low–PD-L1 and high–PD-L1 groups. The TIL grouping factor histomorphometric feature (F3) was found to be statistically significantly (r = −0.56 and P = 0.001) correlated
with QVT1 and QVT6 both referring to the distribution of the most acute angles on the vasculature. (D) The correlation matrix between TIL and QVT features. TIL and QVT
features arranged in rows and columns, respectively.
dropped from 22 to 19 features when we added segmentation-associated imaging, and nondisruptive of normal clinical workflow. The tumor
noise. We additionally measured the performance of our models as vasculature not only was able to predict and monitor tumor behav-
a function of CT slice thickness parameter. As may be observed in ior and response to ICI but also could predict response in patients
the Supplementary Materials, the AUC of QVT classifier decreased treated with the combination of chemotherapy and IO. In future
slightly with increasing slice thickness. work, we will seek to validate QVT in the context of prospective
We do acknowledge, however, that our study did have its limita- biomarker-driven clinical trials.
tions. The size of ICI-treated cohorts both for training and validation
was relatively small. Second, the study was completely retrospective
in nature. Independent prospective validation is needed to show the MATERIALS AND METHODS
prognostic and predictive utility of QVT on patients with NSCLC Datasets
treated with IO. In addition, the association of QVT and TIL features This Health Insurance Portability and Accountability Act of 1996
could only be done on a small subset of cases due to nonavailability (“HIPAA”) regulations–compliant study was approved by the insti-
of tissue for pathomic analysis. The radiogenomic analysis too could tutional review board (IRB 02-13-42C) at the University Hospitals
only be carried out on a different cohort of early-stage patients due Case Medical Center, and the need for informed consent was waived.
to lack of tissue on the ICI cohort for mRNA sequencing due to it A total of 507 NSCLC cases were included in this multisite valida-
being retrospective in nature. tion study to explore the various aspects of our imaging biomarker.
Despite the limitations, QVT could potentially serve as a tool for We studied the association of baseline and delta QVT features
monitoring and predicting response to ICI and help identify patients (defined as the absolute change in QVT features between baseline and
with NSCLC who are likely to benefit from IO. It enjoys several ad- posttreatment scans) with response to therapy and OS of the patients
vantages over currently deployed biomarkers in not only being non- with NSCLC treated with ICI and a combination of ICI and chemo-
invasive but also being inexpensive, derived from routine radiographic therapy. We also studied the molecular, cellular, and radiogenomic
Alilou et al., Sci. Adv. 8, eabq4609 (2022) 25 November 2022 6 of 11
Downloaded
from
https://www.science.org
on
November
28,
2022

## Page 7

SCIENCE ADVANCES | RESEARCH ARTICLE
underpinning of QVT features by exploring the association of QVT associative analysis, we used a subset of D1 (N = 31) cases (digitized
with biological pathways, PD-L1 expression, and TIL density in H&E histology scans of baseline biopsies were only available for 31 cases
images. To predict and monitor treatment response and prognosticating from CCF). A dataset of D6 = 92 patients with early-stage NSCLC from
outcome in ICI-treated patients, N = 162 contrast CT scans of pa- The Cancer Imaging Archive (TCIA) with available RNA sequenc-
tients with advanced NSCLC before and after two to three cycles of ing data was included for the radiogenomic analysis. The prognostic
PD-1/PD-L1 ICI therapy (nivolumab/pembrolizumab/atezolizumab) and predictive potential of QVT biomarkers for predicting response
were included. Samples in which the board-certified radiologist in a combination of ICI and chemotherapy was evaluated on a set
could not isolate a measurable pulmonary nodule on CT scans or of D7 = 45 patients with nonsquamous NSCLC who underwent
the CTs with poor image quality were excluded. The ICI-treated co- ICI (pembrolizumab) and chemotherapy (pemetrexed) between
hort from three institutions was divided into four subsets including October 2015 and August 2018 at CCF. Figure 5 illustrates the data in-
D1 = 62 for training and D2 = 50, D3 = 27, and D4 = 23 for validation. clusion strategy for the various experiments that comprised this study.
A set of 112 patients from January 2012 to August 2017 at the Cleveland Demographics and clinical variables
Clinic Foundation (CCF) was included consecutively, and patients Eastern Cooperative Oncology Group performance status and tumor
were divided to D1 = 62 for training and D2 = 50 for internal valida- node metastasis stage and clinical staging per the American Joint
tion set. Moreover, D3 = 27 patients continuously admitted from Committee on Cancer staging system were used in this study along-
2014 to 2017 at the University of Pennsylvania Health System were side clinical variables including age, sex, and tumor histology. All
identified and used as the first independent test set. In addition, patients (except D5 and D6) included in this study had metastasis
D4 = 23 patients from 2018 to 2020 at the University Hospitals and so were classified into stage IV. Demographics and clinical char-
Cleveland Medical Center were included in this study as the second acteristics for patients were available for D1, D2, D3, and D4 datasets
independent test set. All patients underwent a baseline contrast CT and are summarized in table S5. None of the following clinical fea-
scan before starting treatment with ICIs. Posttreatment scans were tures including gender, race, smoking status, histology subtype, and
available only for D1, D2, and D3. epidermal growth factor receptor mutation status were found to be
An overwhelming majority of the cohorts of patients were treated prognostic of OS.
in an era when ICIs were approved only in the second-line setting at
which point PD-L1 expression was not routinely performed for Image acquisition, nodule detection, and
all patients with NSCLC. Because the prescription of PD-1/PD-L1 vasculature segmentation
inhibitors in this setting does not mandate PD-L1 quantification CT scans were acquired from all ICI-treated patients at baseline
(except pembrolizumab), many of the patients in these cohorts re- and immediately after two to three cycles (6 to 8 weeks) of ICI treat-
ceived nivolumab or atezolizumab without prior PD-L1 testing. To ment for D1, D2, and D3. Scans were acquired using a multislice
assess whether QVT features are associated with PD-L1 expression, (Philips Healthcare, General Electric Health Care, Siemens Health-
a separate cohort of D5 = 204 patients with early-stage NSCLC be- care) CT system with a tube voltage of 100 to 120 kilovolt peak, slice
tween April 2004 and April 2015 with available diagnostic CT scans thickness (spacing) of 1 to 5 mm (mean = 2.82 mm and SD =
from CCF was included in this study. For QVT and TIL density 0.71 mm), and in-plane resolution of 0.75 × 0.75 mm. All CT images
Fig. 5. Data inclusion and experimental workflow.
Alilou et al., Sci. Adv. 8, eabq4609 (2022) 25 November 2022 7 of 11
Downloaded
from
https://www.science.org
on
November
28,
2022

## Page 8

SCIENCE ADVANCES | RESEARCH ARTICLE
were captured with patients in inspiration breath-hold phase after ending points of its centerline. In addition to the initial set of QVT
contrast injection. All scans were acquired using the facilities’ CT features that the authors previously presented (28), the angles of any
chest protocol and standard image reconstruction (15). In the case three consecutive points of the vasculature were measured, and the
of multifocal nodules, the primary nodule was selected according distribution of these angles was dichotomized into 15 bins. We also
to the radiology report at baseline and tracked and delineated in assessed the stability of QVT features in test-retest scans and then
the posttreatment with 3D SLICER software by a board-certified measured their stability against segmentation errors. Additional details
cardiothoracic radiologist (with 8 years of experience). of stability analysis and sensitivity of QVT features to CT parame-
ters are provided in sections SB and SC.
Vascular feature extraction
The manually segmented target nodules were used to compute the Statistical analysis
volume of interest (VOI) and subsequently segment the nodule- Classification
associated vasculature and extract vascular features. Lung regions The primary endpoint of this study was primary clinical response
were automatically isolated from the surrounding anatomy using a defined by response evaluation in solid tumors (RECIST) v1.1. Pa-
multithreshold-based algorithm (29). The vasculature within the lung tients who did not receive ICI after two cycles due to lack of response
regions was segmented from lung parenchyma by applying a vessel or progression as per RECIST were classified as “nonresponders,”
enhancement filter followed by a multithreshold algorithm. The VOI and patients who had radiological response or stable disease as per
is defined as a rectangular prism region that has the nodule in the RECIST and clinical improvement were classified as “responders.” An
center. The size of the VOI is defined relatively with respect to the LDA classifier was trained on D1 with the stable and discriminat-
size of the nodule. A region growing algorithm was used for the seg- ing vascular features to predict the RECIST-based response. Within
mentation of the nodule vasculature (30) within the VOI. A fast- the discovery set D1, the classifier was trained in a threefold cross-
marching algorithm (31) was then used to identify the centerlines of validation setting. The procedure was iterated over 200 runs. The
the 3D segmented vasculature. Figure 6 illustrates the process of performance of the response prediction classifier was assessed with
vasculature segmentation. A set of 74 QVT features were measured the ROC as AUC. In addition, an unsupervised hierarchal clustering
from points, branches, and the entire vasculature centerlines. These analysis (using the clustergram function in MATLAB) was conducted
features pertain to the tortuosity, curvature, and branching statistics on QVT features (15).
as well as the volume of the vasculature. Curvature at a point on the Survival analysis
vascular centerline segment is measured by fitting a circle that ap- The secondary endpoint of this study was OS, which was defined as
proximates the shape of the segment the best. The tortuosity of a the time from the date of the disease diagnosis until the date of
vascular segment is measured as the ratio of its centerline length with death (or until the date that the patient was last known to be alive if
respect to the length of a straight line that connects the starting and censored). The median follow-up of OS posttreatment was 16 months
Fig. 6. The main workflow of QVT feature extraction. (A) Identifying tumor position by a radiologist. (B) Segmentation of nodule and lung regions. (C) Vasculature
segmentation. (D) Identifying nodule-associated vasculature. (E) Extraction of the vessel’s centerlines. (F) Extraction of QVT features from centerlines.
Alilou et al., Sci. Adv. 8, eabq4609 (2022) 25 November 2022 8 of 11
Downloaded
from
https://www.science.org
on
November
28,
2022

## Page 9

SCIENCE ADVANCES | RESEARCH ARTICLE
(range, 1 to 45 months). The Kaplan-Meier survival analysis and version 2 (Illumina, San Diego, CA, USA). The Cancer Genome Atlas
log-rank statistical tests were performed to assess the univariable gene expression data are publicly available for download (39). An em-
discriminative ability of the features on OS (32). The prognostic value pirical analysis using the Wilcoxon rank sum test of the 22,126 genes
of vascular features on OS was estimated by using the QRS. To across the high and low QVT yielded a set of differentially expressed
build the multivariate signature for OS, the least absolute shrinkage genes. The Benjamini and Hochberg method was used to adjust the
and selection operator (LASSO) Cox regression model (33) was used P values and control for the FDR (<0.01). Gene Ontology analysis was
to identify the prognostic features from the subset of 74 stable QVT performed to identify distinct biological processes (40, 41), which
features in the training set (D1). A QRS was computed for each structures and classifies genes on the basis of the known molecular
patient according to a linear combination of selected features with and cellular biological processes and provides the relationship be-
corresponding nonzero coefficients from the LASSO Cox model in tween those processes. These pathways were chosen on the basis of
the training set (D1). On the basis of the cutoff value of QRS on D1, their biological significance in regulating immune response, cell
the patients on D2, D3, and D4 were stratified into high- and low- adhesion, and carcinogenesis. GSEA was applied on major identi-
risk groups. A multivariable Cox proportional hazards model was fied biological processes/pathways to determine separate enrich-
used to evaluate the ability of the QRS in predicting OS. In addition, ment scores for each pairing of a sample and gene set (42). The lists
relative HRs with 95% CI were calculated. The median follow-up of genes involved in each pathway were obtained from the Molecular
was also estimated with the reverse Kaplan-Meier method (15, 34). Signatures Database. Last, a pairwise Wilcoxon rank sum test on
Association of QVT with PD-L1 expression enrichment scores was performed across high– and low–QVT feature
Correlation analysis of QVT features with PD-L1 expression was groups to obtain the strength of association between the pathway
also performed. In this regard, PD-L1 > 50% was used as cutoff value enrichment score and the feature values.
to divide patients in D5 into PD-L1low and PD-L1high groups. The
Wilcoxon rank sum significance test was then performed on QVT
SUPPLEMENTARY MATERIALS
features between PD-L1low and PD-L1high groups to evaluate whether
Supplementary material for this article is available at https://science.org/doi/10.1126/
there were significant differences between QVT feature and PD-L1 sciadv.abq4609
level expression. All tests were two-sided, and P values less than View/request a protocol for this paper from Bio-protocol.
0.05 were considered statistically significant.
Association of QVT with TIL density on digital REFERENCES AND NOTES
pathology images 1. H. Borghaei, L. Paz-Ares, L. Horn, D. R. Spigel, M. Steins, N. E. Ready, L. Q. Chow,
For QVT and TIL density associative analysis on subset of 31 cases E. E. Vokes, E. Felip, E. Holgado, F. Barlesi, M. Kohlhäufl, O. Arrieta, M. A. Burgio, J. Fayette,
from D1, we used an automated detection of TILs in H&E images (35) H. Lena, E. Poddubskaya, D. E. Gerber, S. N. Gettinger, C. M. Rudin, N. Rizvi, L. Crinò,
G. R. Blumenschein Jr., S. J. Antonia, C. Dorange, C. T. Harbison, F. Graf Finckenstein,
followed by computational spatial clustering metrics. A watershed-
J. R. Brahmer, Nivolumab versus docetaxel in advanced nonsquamous non–small-cell
based algorithm (36) was first applied to segment nuclei on the lung cancer. N. Engl. J. Med. 373, 1627–1639 (2015).
image. Considering that lymphocyte nuclei are generally distin- 2. M. Reck, D. Rodríguez-Abreu, A. G. Robinson, R. Hui, T. Csőszi, A. Fülöp, M. Gottfried,
guished from other cell nuclei by their smaller size, more rounded N. Peled, A. Tafreshi, S. Cuffe, M. O'Brien, S. Rao, K. Hotta, M. A. Leiby, G. M. Lubiniecki,
shape, and a darker homogeneous staining, we classified the seg- Y. Shentu, R. Rangwala, J. R. Brahmer, KEYNOTE-024 Investigators, Pembrolizumab versus
chemotherapy for PD-L1–positive non–small-cell lung cancer. N. Engl. J. Med. 375,
mented nuclei into either lymphocytes or nonlymphocytes (mainly,
1823–1833 (2016).
tumor cells) using nuclei texture, shape, and color features (37). 3. D. M. Pardoll, The blockade of immune checkpoints in cancer immunotherapy. Nat. Rev.
Twelve features quantifying the density or compactness of TILs Cancer 12, 252–264 (2012).
were extracted from the surgical specimens. Each lymphocyte is 4. H. Linardou, H. Gogas, Toxicity management of immunotherapy for patients
with metastatic melanoma. Ann. Transl. Med. 4, 272 (2016).
characterized by its own local morphological feature and by a set of
5. L. Spain, G. Walls, M. Julve, K. O’Meara, T. Schmid, E. Kalaitzaki, S. Turajlic, M. Gore, J. Rees,
contextual features to describe the lymphocyte and its neighbor-
J. Larkin, Neurotoxicity from immune-checkpoint inhibition in the treatment
hood. Lymphocytes are grouped under a Dirichlet process Gaussian of melanoma: A single centre experience and review of the literature. Ann. Oncol. 28,
mixture model, which involves clustering the data via a non- 377–385 (2017).
parametric Bayesian framework that describes distributions over 6. S. P. Patel, R. Kurzrock, PD-L1 expression as a predictive biomarker in cancer
immunotherapy. Mol. Cancer Ther. 14, 847–856 (2015).
mixture models with an infinite number of mixture components.
7. W. J. Lesterhuis, J. B. A. G. Haanen, C. J. A. Punt, Cancer immunotherapy—Revisited.
The advantage of such grouping is that one does not need to make Nat. Rev. Drug Discov. 10, 591–600 (2011).
any assumptions about the number of TIL clusters. Each image is 8. B. Thom, M. Mamoor, J. A. Lavery, S. S. Baxi, N. Khan, L. J. Rogak, R. Sidlow, D. Korenstein,
then characterized by the histogram of occurrences of the iden- The experience of financial toxicity among advanced melanoma patients treated
with immunotherapy. J. Psychosoc. Oncol. 39, 285–293 (2021).
tified TILs within the particular partition defined by the groups
9. L. B. Kennedy, A. K. S. Salama, A review of cancer immunotherapy toxicity. CA Cancer
(15). Details regarding the extracted features are provided in (35). J. Clin. 70, 86–104 (2020).
To investigate the QVT-TIL associations, a pairwise Spearman 10. R. Li, D. Han, J. Shi, Y. X. Han, P. Tan, R. Zhang, J. Li, Choosing tumor mutational burden
correlation was performed between each of the top QVT and TIL wisely for immunotherapy: A hard road to explore. Biochim. Biophys. Acta Rev. Cancer
compactness measures followed by Benjamini-Hochberg method 1874, 188420 (2020).
11. S. Trebeschi, I. Kurilova, A. M. Călin, D. M. J. Lambregts, E. F. Smit, H. Aerts, R. G. H. Beets-Tan,
(38) to adjust the P values and control for the false discovery rate
Radiomic biomarkers for the prediction of immunotherapy outcome in patients
(FDR; <0.01). with metastatic non-small cell lung cancer. J. Clin. Oncol. 35, e14520 (2017).
Association with GSEA pathways 12. C. Tang, B. Hobbs, A. Amer, X. Li, C. Behrens, J. R. Canales, E. P. Cuentas, P. Villalobos,
A dataset of D6 = 92 patients with early-stage NSCLC from TCIA D. Fried, J. Y. Chang, D. S. Hong, J. W. Welsh, B. Sepesi, L. Court, I. I. Wistuba, E. J. Koay,
Development of an immune-pathology informed radiomics model for non-small cell
with available mRNA sequencing data was included for radiogenomic
lung cancer. Sci. Rep. 8, 1922 (2018).
analysis. Radiogenomic analysis was performed using mRNA se-
13. D. Saeed-Vafaa, R. Bravob, J. A. Dean, A. El-Kenawic, Nathaniel, M. Père, M. Strobla,
quencing data obtained with Illumina Genome Analyzer Sequencing C. Daniels, O. Stringfieldd, M. Damaghid, Ilke, Tunalid, L. V. Browna, L. Curtin, D. Nicholb,
Alilou et al., Sci. Adv. 8, eabq4609 (2022) 25 November 2022 9 of 11
Downloaded
from
https://www.science.org
on
November
28,
2022

## Page 10

SCIENCE ADVANCES | RESEARCH ARTICLE
H. Peck, R. J. Gilliesd, J. A. Gallaherb, Combining radiomics and mathematical modeling to platinum-based chemotherapy and is prognostic of overall survival in small cell lung
to elucidate mechanisms of resistance to immune checkpoint blockade in non-small cell cancer. Front. Oncol. 11, 744724 (2021).
lung cancer. bioRxiv 10.1101/190561 (2017). 33. J. Fan, R. Li, Variable selection for Cox’s proportional hazards model and frailty model.
14. R. Sun, E. Limkin, M. Vakalopoulou, L. Dercle, S. Champiat, S. Han, L. Verlingue, Ann. Statist. 30, 74–99 (2002).
D. Brandao, A. Lancia, S. Ammari, A. Hollebecque, J. Scoazec, A. Marabelle, C. Massard, 34. K. Jazieh, M. Khorrami, A. Saad, M. Gad, A. Gupta, P. Patil, V. S. Viswanathan, P. Rajiah,
J. Soria, C. Robert, N. Paragios, E. Deutsch, C. Ferté, A radiomics approach to assess C. J. Nock, M. Gilkey, P. Fu, N. A. Pennell, A. Madabhushi, Novel imaging biomarkers
tumour-infiltrating CD8 cells and response to anti-PD-1 or anti-PD-L1 immunotherapy: predict outcomes in stage III unresectable non-small cell lung cancer treated
An imaging biomarker, retrospective multicohort study. Lancet Oncol. 19, 1180–1191 with chemoradiation and durvalumab. J. Immunother. Cancer 10, e003778 (2022).
(2018). 35. G. Corredor, X. Wang, Y. Zhou, C. Lu, P. Fu, K. Syrigos, D. L. Rimm, M. Yang, E. Romero,
15. M. Khorrami, P. Prasanna, A. Gupta, P. Patil, P. D. Velu, R. Thawani, G. Corredor, M. Alilou, K. A. Schalper, V. Velcheti, A. Madabhushi, Spatial architecture and arrangement
K. Bera, P. Fu, M. Feldman, V. Velcheti, A. Madabhushi, Changes in CT radiomic features of tumor-infiltrating lymphocytes for predicting likelihood of recurrence in early-stage
associated with lymphocyte distribution predict overall survival and response non-small cell lung cancer. Clin. Cancer Res. 25, 1526–1534 (2019).
to immunotherapy in non-small cell lung cancer. Cancer Immunol. Res. 8, 108–119 (2019). 36. L. Vincent, P. Soille, Watersheds in digital spaces: An efficient algorithm based
16. S. Trebeschi, S. G. Drago, N. J. Birkbak, I. Kurilova, A. M. Cǎlin, A. Delli Pizzi, F. Lalezari, on immersion simulations. IEEE Trans. Pattern Anal. Mach. Intell. 13, 583–598 (1991).
D. M. J. Lambregts, M. W. Rohaan, C. Parmar, E. A. Rozeman, K. J. Hartemink, C. Swanton, 37. F. Xing, L. Yang, Robust nucleus/cell detection and segmentation in digital pathology
J. B. A. G. Haanen, C. U. Blank, E. F. Smit, R. G. H. Beets-Tan, H. J. W. L. Aerts, Predicting and microscopy images: A comprehensive review. IEEE Rev. Biomed. Eng. 9, 234–263 (2016).
response to cancer immunotherapy using noninvasive radiomic biomarkers. Ann. Oncol. 38. J. A. Ferreira, A. H. Zwinderman, On the Benjamini–Hochberg method. Ann. Statist. 34,
30, 998–1004 (2019).
1827–1849 (2006).
17. S. A. Hendry, R. H. Farnsworth, B. Solomon, M. G. Achen, S. A. Stacker, S. B. Fox, The role
39. S. Bakr, O. Gevaert, S. Echegaray, K. Ayers, M. Zhou, M. Shafiq, H. Zheng, J. A. Benson,
of the tumor vasculature in the host immune response: Implications for therapeutic
W. Zhang, A. N. C. Leung, M. Kadoch, C. D. Hoang, J. Shrager, A. Quon, D. L. Rubin,
strategies targeting the tumor microenvironment. Front. Immunol. 7, 621 (2016). S. K. Plevritis, S. Napel, A radiogenomic dataset of non-small cell lung cancer. Sci. Data 5,
18. M. O. Li, Y. Y. Wan, S. Sanjabi, A. K. L. Robertson, R. A. Flavell, Transforming growth
180202 (2018).
factor- regulation of immune responses. Annu. Rev. Immunol. 24, 99–146 (2006).
40. M. Ashburner, C. A. Ball, J. A. Blake, D. Botstein, H. Butler, J. M. Cherry, A. P. Davis,
19. J. Duan, Y. Wang, S. Jiao, Checkpoint blockade-based immunotherapy in the context
K. Dolinski, S. S. Dwight, J. T. Eppig, M. A. Harris, D. P. Hill, L. Issel-Tarver, A. Kasarskis,
of tumor microenvironment: Opportunities and challenges. Cancer Med. 7, 4517–4529
S. Lewis, J. C. Matese, J. E. Richardson, M. Ringwald, G. M. Rubin, G. Sherlock, Gene
(2018). ontology: Tool for the unification of biology. The Gene Ontology Consortium. Nat. Genet.
20. R. K. Jain, Normalizing tumor vasculature with anti-angiogenic therapy: A new paradigm 25, 25–29 (2000).
for combination therapy. Nat. Med. 7, 987–989 (2001).
41. The Gene Ontology Consortium, Expansion of the gene ontology knowledgebase
21. T. Stylianopoulos, L. L. Munn, R. K. Jain, Reengineering the physical microenvironment
and resources. Nucleic Acids Res. 45, D331–D338 (2017).
of tumors to improve drug delivery and efficacy: From mathematical modeling to bench
to bedside. Trends Cancer 4, 292–319 (2018). 42. A. Subramanian, P. Tamayo, V. K. Mootha, S. Mukherjee, B. L. Ebert, M. A. Gillette,
A. Paulovich, S. L. Pomeroy, T. R. Golub, E. S. Lander, J. P. Mesirov, Gene set enrichment
22. M. J. Cross, L. Claesson-Welsh, FGF and VEGF function in angiogenesis: Signalling
pathways, biological responses and therapeutic inhibition. Trends Pharmacol. Sci. 22, analysis: A knowledge-based approach for interpreting genome-wide expression
profiles. Proc. Natl. Acad. Sci. U.S.A. 102, 15545–15550 (2005).
201–207 (2001).
23. S. Xue, M. Hu, P. Li, J. Ma, L. Xie, F. Teng, Y. Zhu, B. Fan, D. Mu, J. Yu, Relationship between
expression of PD-L1 and tumor angiogenesis, proliferation, and invasion in glioma. Acknowledgments
Oncotarget 8, 49702–49712 (2017). Funding: Research reported in this publication was supported by the National Cancer
24. Y. L. Zhang, J. Li, H. Y. Mo, F. Qiu, L. M. Zheng, C. N. Qian, Y. X. Zeng, Different subsets Institute under award numbers 1U24CA199374-01, R01CA249992-01A1, R01CA202752-01A1,
of tumor infiltrating lymphocytes correlate with NPC progression in different ways. R01CA208236-01A1, R01CA216579-01A1, R01CA220581-01A1, R01CA257612-01A1,
Mol. Cancer 9, 4 (2010). 1U01CA239055-01, 1U01CA248226-01, and 1U54CA254566-01; National Heart, Lung, and
25. J. Goc, C. Germain, T. K. D. Vo-Bourgais, A. Lupo, C. Klein, S. Knockaert, L. de Chaisemartin, Blood Institute, 1R01HL15127701A1; National Institute of Biomedical Imaging and
H. Ouakrim, E. Becht, M. Alifano, P. Validire, R. Remark, S. A. Hammond, I. Cremer, Bioengineering, 1R43EB028736-01; National Center for Research Resources under award
D. Damotte, W. H. Fridman, C. Sautès-Fridman, M. C. Dieu-Nosjean, Dendritic cells number 1 C06 RR12463-01, VA Merit Review Award IBX004121A from the U.S. Department of
in tumor-associated tertiary lymphoid structures signal a Th1 cytotoxic immune Veterans Affairs Biomedical Laboratory Research and Development Service the Office of the
contexture and license the positive prognostic value of infiltrating CD8+ T cells. Cancer Res. Assistant Secretary of Defense for Health Affairs, through the Breast Cancer Research Program
74, 705–715 (2014). (W81XWH-19-1-0668) and the Prostate Cancer Research Program (W81XWH-15-1-0558 and
26. R. Ganss, D. Hanahan, Tumor microenvironment can restrict the effectiveness of activated W81XWH-20-1-0851); the Lung Cancer Research Program (W81XWH-18-1-0440 and
antitumor lymphocytes. Cancer Res. 58, 4673–4681 (1998). W81XWH-20-1-0595); the Peer Reviewed Cancer Research Program (W81XWH-18-1-0404); the
27. J. J. Luke, R. Bao, R. F. Sweis, S. Spranger, T. F. Gajewski, WNT/-catenin pathway Kidney Precision Medicine Project (KPMP) Glue Grant; the Ohio Third Frontier Technology
activation correlates with immune exclusion across human cancers. Clin. Cancer Res. 25, Validation Fund; the Clinical and Translational Science Collaborative of Cleveland
(UL1TR0002548) from the National Center for Advancing Translational Sciences (NCATS)
3074–3083 (2019).
component of the National Institutes of Health (NIH) and NIH roadmap for Medical Research;
28. M. Alilou, M. Orooji, N. Beig, P. Prasanna, P. Rajiah, C. Donatelli, V. Velcheti, S. Rakshit,
and the Wallace H. Coulter Foundation Program in the Department of Biomedical Engineering
M. Yang, F. Jacono, R. Gilkeson, P. Linden, A. Madabhushi, Quantitative vessel tortuosity:
at Case Western Reserve University. The content is solely the responsibility of the authors and
A potential CT imaging biomarker for distinguishing lung granulomas
does not necessarily represent the official views of the NIH, the U.S. Department of Veterans
from adenocarcinomas. Sci. Rep. 8, 15290 (2018).
Affairs, the Department of Defense, or the U.S. government. The funders had no role in data
29. S. Hu, E. A. Hoffman, J. M. Reinhardt, Automatic lung segmentation for accurate collection, data analysis, data interpretation, or preparing the results. Author contributions:
quantitation of volumetric x-ray CT images. IEEE Trans. Med. Imaging 20, 490–498 Conceptualization: M.A., M.K., P.Pr., and A.M. Methodology: M.A., M.K., P.Pr., and A.M. Software:
(2001). M.A. and P.Pr. Validation: M.A., M.K., P.Pr., and A.M. Formal analysis: M.A., M.K., P.Pr., and P.F.
30. R. D. Rudyanto, S. Kerkstra, E. M. van Rikxoort, C. Fetita, P. Y. Brillet, C. Lefevre, W. Xue, Investigation: M.A. and P.Pr. Resources: M.A. Data curation: K.B., A.G., V.S.V., P.Pa., P.D.V., and
X. Zhu, J. Liang, İ. Öksüz, D. Ünay, K. Kadipaşaogˇlu, R. S. J. Estépar, J. C. Ross, G. R. Washko, V.V. Writing—original draft preparation: M.A., M.K., P.Pr., and P.Pa. Writing—review and
J. C. Prieto, M. H. Hoyos, M. Orkisz, H. Meine, M. Hüllebrand, C. Stöcker, F. L. Mir, editing: M.K., M.A., A.G., V.V., and A.M. Visualization: M.A. Supervision: V.V. and A.M. All authors
V. Naranjo, E. Villanueva, M. Staring, C. Xiao, B. C. Stoel, A. Fabijanska, E. Smistad, have read and agreed to the published version of the manuscript. All authors confirm that
A. C. Elster, F. Lindseth, A. H. Foruzan, R. Kiros, K. Popuri, D. Cobzas, D. Jimenez-Carretero, they had full access to all the data in the study and accept responsibility to submit for
A. Santos, M. J. Ledesma-Carbayo, M. Helmberger, M. Urschler, M. Pienn, publication. A.M. acts as the guarantors of the study. Competing interests: A.M. is an equity
D. G. H. Bosboom, A. Campo, M. Prokop, P. A. de Jong, C. Ortiz-de-Solorzano, holder in Elucid Bioimaging and Inspirata Inc. In addition, he has served as a scientific advisory
A. Muñoz-Barrutia, B. van Ginneken, Comparing algorithms for automated vessel board member for Inspirata Inc., Astrazeneca, Bristol Meyers-Squibb, and Merck. Now, he
segmentation in computed tomography scans of the lung: The VESSEL12 study. Med. serves on the advisory board of Aiforia Inc. He also has sponsored research agreements with
Image Anal. 18, 1217–1232 (2014). Philips, AstraZeneca, and Bristol Meyers-Squibb. His technology has been licensed to Elucid
31. J. A. Sethian, Fast marching methods. SIAM Rev. 41, 199–235 (1999). Bioimaging. He is also involved in an NIH U24 grant with PathCore Inc. and three different
32. P. Jain, M. Khorrami, A. Gupta, P. Rajiah, K. Bera, V. S. Viswanathan, P. Fu, A. Dowlati, R01 grants with Inspirata Inc. P.Pa. has a financial and advisory relationship with Astrazeneca
A. Madabhushi, Novel non-invasive radiomic signature on CT scans predicts response and Jazz Pharmaceuticals. A.M. and P.Pr. are inventors on a patent related to this work filed
Alilou et al., Sci. Adv. 8, eabq4609 (2022) 25 November 2022 10 of 11
Downloaded
from
https://www.science.org
on
November
28,
2022

## Page 11

SCIENCE ADVANCES | RESEARCH ARTICLE
by Case Western Reserve University (no. US9483822B2, filed 28 January 2015, published Supplementary Materials. The dataset D6 used in this study is publicly available open source
1 November 2016). A.M. is an inventor on four patents related to this work filed by Case Western and can be accessed through the corresponding sources: https://wiki.cancerimagingarchive.
Reserve University (no. US9767555B2, filed 10 December 2015, published 19 September 2017; net/display/Public/NSCLC+Radiogenomics#28672347a99a795ff4454409862a398ffc076b98.
no. US9984462B2, filed 15 September 2017, published 29 May 2018; no. US10004471B2, filed Code, model files, and extra software used in this manuscript to reproduce the results are
2 August 2016, published 26 June 2018; and no. US10398399B2, filed 27 March 2018). A.M. available at https://zenodo.org/record/7301761.
and M.A. are inventors on a patent related to this work filed by Case Western Reserve
University (no. US10064594B2, filed 2 August 2016), published 4 September 2018). A.M., M.A.,
and V.V. are inventors on a patent related to this work filed by Case Western Reserve University Submitted 9 April 2022
(no. US10441215B2, filed 9 February 2018, published 15 October 2019). The authors declare Accepted 6 October 2022
that they have no other competing interests. Data and materials availability: All data Published 25 November 2022
needed to evaluate the conclusions in the paper are present in the paper and/or the 10.1126/sciadv.abq4609
Alilou et al., Sci. Adv. 8, eabq4609 (2022) 25 November 2022 11 of 11
Downloaded
from
https://www.science.org
on
November
28,
2022

## Page 12

A tumor vasculature–based imaging biomarker for predicting response and
survival in patients with lung cancer treated with checkpoint inhibitors
Mehdi AlilouMohammadhadi KhorramiPrateek PrasannaKaustav BeraAmit GuptaVidya Sankar ViswanathanPradnya
PatilPriya Darsini VeluPingfu FuVamsidhar VelchetiAnant Madabhushi
Sci. Adv., 8 (47), eabq4609. • DOI: 10.1126/sciadv.abq4609
View the article online
https://www.science.org/doi/10.1126/sciadv.abq4609
Permissions
https://www.science.org/help/reprints-and-permissions
Use of this article is subject to the Terms of service
Science Advances (ISSN ) is published by the American Association for the Advancement of Science. 1200 New York Avenue NW,
Washington, DC 20005. The title Science Advances is a registered trademark of AAAS.
Copyright © 2022 The Authors, some rights reserved; exclusive licensee American Association for the Advancement of Science. No claim
to original U.S. Government Works. Distributed under a Creative Commons Attribution NonCommercial License 4.0 (CC BY-NC).
Downloaded
from
https://www.science.org
on
November
28,
2022
