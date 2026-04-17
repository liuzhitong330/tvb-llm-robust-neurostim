# Top 20 Research Ideas: Epilepsy × Computational Modeling

Ranked by combined score (Novelty + TVB Feasibility + Clinical Impact, max 30).
Judge model: `claude-opus-4-5` · Grouping model: `claude-haiku-4-5`

---

## #1 — Digital twins and multimodal AI for precision neurology
**Score: 25/30** &nbsp;|&nbsp; Novelty: 7/10 &nbsp;·&nbsp; TVB Feasibility: 10/10 &nbsp;·&nbsp; Clinical Impact: 8/10

**Summary:** Integrating multi-omic data, neuroimaging, computational simulations, and AI systems to create patient-specific digital models for personalized diagnosis, prognosis, and treatment planning.

**Representative idea:** _Digital twin systems integrating multi-modal data for personalized epilepsy diagnosis and treatment planning_

**Key opportunity:** Establish TVB as the simulation backbone of a digital twin platform that ingests imaging, EEG, and genetics to optimize individual stimulation parameters.

**Rationale:** Digital twin terminology is trending but true multi-omic integration remains largely aspirational; substantial innovation space exists. TVB is fundamentally a digital twin platform and the natural core for such integration. Clinical impact is high if validated, enabling personalized treatment optimization.

**Source papers:** [PMID:19](https://pubmed.ncbi.nlm.nih.gov/19/) · [PMID:41](https://pubmed.ncbi.nlm.nih.gov/41/) · [PMID:100](https://pubmed.ncbi.nlm.nih.gov/100/) · [PMID:172](https://pubmed.ncbi.nlm.nih.gov/172/)

_✅ TVB-addressable · 4 source ideas_

---

## #2 — Computational biomarkers for neuromodulation outcome prediction
**Score: 25/30** &nbsp;|&nbsp; Novelty: 7/10 &nbsp;·&nbsp; TVB Feasibility: 9/10 &nbsp;·&nbsp; Clinical Impact: 9/10

**Summary:** Identifying quantitative computational and electrophysiological biomarkers predicting surgical outcomes, RNS responses, and DBS efficacy using network analysis and machine learning.

**Representative idea:** _Neural network excitability index (NNEI) and computational biomarkers predicting neuromodulation outcomes_

**Key opportunity:** Prospectively validate a TVB-derived network excitability index against RNS/DBS outcomes in a multi-center registry.

**Rationale:** Quantitative pre-surgical biomarkers predicting DBS/RNS response are urgently needed but lack validation; current patient selection relies heavily on clinical judgment. TVB is ideally suited for deriving network-based excitability indices from patient connectomes. Direct clinical translation: better patient selection reduces failed surgeries.

**Source papers:** [PMID:248](https://pubmed.ncbi.nlm.nih.gov/248/) · [PMID:269](https://pubmed.ncbi.nlm.nih.gov/269/) · [PMID:270](https://pubmed.ncbi.nlm.nih.gov/270/) · [PMID:273](https://pubmed.ncbi.nlm.nih.gov/273/) · [PMID:275](https://pubmed.ncbi.nlm.nih.gov/275/)

_✅ TVB-addressable · 5 source ideas_

---

## #3 — Optimal stimulation target identification via network analysis
**Score: 24/30** &nbsp;|&nbsp; Novelty: 7/10 &nbsp;·&nbsp; TVB Feasibility: 9/10 &nbsp;·&nbsp; Clinical Impact: 8/10

**Summary:** Using connectivity metrics, network influence measures, and circuit-based approaches to identify optimal brain stimulation locations beyond seizure onset zone, with focus on node centrality and effective connectivity.

**Representative idea:** _AI and circuit-based approaches to identify optimal stimulation locations based on network influence rather than seizure onset zone alone_

**Key opportunity:** Retrospective analysis correlating TVB-predicted optimal targets with actual responder outcomes across multiple centers.

**Rationale:** Moving beyond seizure onset zone to network-level targets is conceptually emerging but underexplored empirically. TVB's whole-brain connectivity modeling directly supports in-silico target screening and network perturbation analysis. High clinical relevance for improving RNS/DBS responder rates.

**Source papers:** [PMID:3](https://pubmed.ncbi.nlm.nih.gov/3/) · [PMID:72](https://pubmed.ncbi.nlm.nih.gov/72/) · [PMID:174](https://pubmed.ncbi.nlm.nih.gov/174/) · [PMID:220](https://pubmed.ncbi.nlm.nih.gov/220/) · [PMID:221](https://pubmed.ncbi.nlm.nih.gov/221/)

_✅ TVB-addressable · 6 source ideas_

---

## #4 — Personalized whole-brain computational models for epilepsy
**Score: 23/30** &nbsp;|&nbsp; Novelty: 5/10 &nbsp;·&nbsp; TVB Feasibility: 10/10 &nbsp;·&nbsp; Clinical Impact: 8/10

**Summary:** Developing and validating patient-specific neural mass and network models to predict seizure propagation, optimize surgical targeting, and guide neuromodulation strategies with integration of structural connectivity and lesion information.

**Representative idea:** _Patient-specific whole-brain computational models for predicting seizure propagation patterns and optimizing surgical targeting_

**Key opportunity:** Prospective multicenter validation trial comparing TVB-guided versus standard surgical planning outcomes.

**Rationale:** This is TVB's core use case with active research by multiple groups (Jirsa, Proix, Saggio), limiting novelty. However, TVB is purpose-built for this application with established pipelines. Clinical impact is high given ongoing surgical planning studies, though prospective validation remains incomplete.

**Source papers:** [PMID:11](https://pubmed.ncbi.nlm.nih.gov/11/) · [PMID:39](https://pubmed.ncbi.nlm.nih.gov/39/) · [PMID:81](https://pubmed.ncbi.nlm.nih.gov/81/) · [PMID:83](https://pubmed.ncbi.nlm.nih.gov/83/) · [PMID:135](https://pubmed.ncbi.nlm.nih.gov/135/)

_✅ TVB-addressable · 8 source ideas_

---

## #5 — Closed-loop adaptive stimulation parameter optimization
**Score: 23/30** &nbsp;|&nbsp; Novelty: 7/10 &nbsp;·&nbsp; TVB Feasibility: 7/10 &nbsp;·&nbsp; Clinical Impact: 9/10

**Summary:** Designing real-time systems that dynamically adjust stimulation parameters based on brain state estimation and physiological feedback, including reinforcement learning and control-theoretic approaches.

**Representative idea:** _Closed-loop systems dynamically adjusting stimulation parameters based on real-time brain state and individual physiology_

**Key opportunity:** Use TVB digital twins to pre-train reinforcement learning controllers for subsequent transfer to implanted devices.

**Rationale:** Current RNS uses fixed detection-triggered responses; true state-adaptive optimization is nascent. TVB can serve as a digital twin for offline policy training and controller design, though real-time implementation requires separate systems. Transformative potential for treatment personalization.

**Source papers:** [PMID:5](https://pubmed.ncbi.nlm.nih.gov/5/) · [PMID:76](https://pubmed.ncbi.nlm.nih.gov/76/) · [PMID:105](https://pubmed.ncbi.nlm.nih.gov/105/) · [PMID:246](https://pubmed.ncbi.nlm.nih.gov/246/) · [PMID:254](https://pubmed.ncbi.nlm.nih.gov/254/)

_✅ TVB-addressable · 7 source ideas_

---

## #6 — Patient-specific stimulation waveform and parameter optimization
**Score: 22/30** &nbsp;|&nbsp; Novelty: 7/10 &nbsp;·&nbsp; TVB Feasibility: 7/10 &nbsp;·&nbsp; Clinical Impact: 8/10

**Summary:** Developing methods to optimize stimulation parameters and complex waveforms at the individual level through computational modeling, artifact-free recording, and real-time adaptive approaches.

**Representative idea:** _Patient-specific stimulation waveform optimization including chaos-rhythm-based and complex polyphasic approaches_

**Key opportunity:** Build closed-loop TVB simulation framework that tests thousands of waveform parameters in silico before clinical implementation.

**Rationale:** Complex waveform optimization beyond standard parameters is underexplored; chaos-rhythm and polyphasic approaches are genuinely novel. TVB can simulate network responses to varied stimulation protocols for patient-specific optimization. High clinical potential given variable stimulation response rates.

**Source papers:** [PMID:43](https://pubmed.ncbi.nlm.nih.gov/43/) · [PMID:255](https://pubmed.ncbi.nlm.nih.gov/255/) · [PMID:256](https://pubmed.ncbi.nlm.nih.gov/256/) · [PMID:263](https://pubmed.ncbi.nlm.nih.gov/263/) · [PMID:273](https://pubmed.ncbi.nlm.nih.gov/273/)

_✅ TVB-addressable · 5 source ideas_

---

## #7 — Seizure network mapping and propagation analysis
**Score: 22/30** &nbsp;|&nbsp; Novelty: 5/10 &nbsp;·&nbsp; TVB Feasibility: 9/10 &nbsp;·&nbsp; Clinical Impact: 8/10

**Summary:** Identifying epileptogenic networks, analyzing seizure propagation pathways, characterizing secondary foci, and mapping multi-lobar involvement using functional and structural imaging.

**Representative idea:** _Seizure network mapping identifying epileptogenic networks and secondary foci for improved surgical targeting_

**Key opportunity:** Develop TVB-based preoperative simulation of seizure propagation to predict which secondary foci require resection versus spare.

**Rationale:** Network-based seizure mapping is established but substantial gaps remain in identifying secondary foci and multi-lobar spread. TVB was specifically designed for this purpose using patient connectomes and is the premier tool for propagation modeling. Clinical impact is high for improving surgical targeting in complex cases.

**Source papers:** [PMID:42](https://pubmed.ncbi.nlm.nih.gov/42/) · [PMID:68](https://pubmed.ncbi.nlm.nih.gov/68/) · [PMID:157](https://pubmed.ncbi.nlm.nih.gov/157/) · [PMID:200](https://pubmed.ncbi.nlm.nih.gov/200/) · [PMID:222](https://pubmed.ncbi.nlm.nih.gov/222/)

_✅ TVB-addressable · 5 source ideas_

---

## #8 — Thalamic stimulation for seizure control and epilepsy subtypes
**Score: 22/30** &nbsp;|&nbsp; Novelty: 6/10 &nbsp;·&nbsp; TVB Feasibility: 8/10 &nbsp;·&nbsp; Clinical Impact: 8/10

**Summary:** Characterizing anterior and centromedian thalamic stimulation mechanisms, optimizing parameters for absence and neocortical seizures, and investigating thalamic engagement variability.

**Representative idea:** _Thalamic DBS mechanisms and optimization for absence seizures and neocortical epilepsy control_

**Key opportunity:** Build a TVB-based thalamic stimulation parameter optimization pipeline validated against existing ANT-DBS outcome datasets.

**Rationale:** Thalamic DBS mechanisms remain incompletely understood despite clinical use; subtype-specific optimization is a clear gap. TVB excels at modeling thalamocortical loops and can simulate parameter variations across patient-specific anatomies. High clinical impact given growing DBS adoption for drug-resistant cases.

**Source papers:** [PMID:8](https://pubmed.ncbi.nlm.nih.gov/8/) · [PMID:40](https://pubmed.ncbi.nlm.nih.gov/40/) · [PMID:80](https://pubmed.ncbi.nlm.nih.gov/80/) · [PMID:115](https://pubmed.ncbi.nlm.nih.gov/115/) · [PMID:116](https://pubmed.ncbi.nlm.nih.gov/116/)

_✅ TVB-addressable · 6 source ideas_

---

## #9 — Responsive neurostimulation (RNS) long-term efficacy and mechanisms
**Score: 21/30** &nbsp;|&nbsp; Novelty: 6/10 &nbsp;·&nbsp; TVB Feasibility: 6/10 &nbsp;·&nbsp; Clinical Impact: 9/10

**Summary:** Investigating mechanisms of RNS-induced seizure suppression, identifying biomarkers predicting long-term responder status, and characterizing network effects of electrode placement effectiveness.

**Representative idea:** _RNS long-term efficacy trajectory mechanisms and biomarkers predicting sustained seizure control durability_

**Key opportunity:** Longitudinal TVB modeling of network reorganization trajectories in RNS responders versus non-responders.

**Rationale:** RNS mechanisms remain poorly understood despite 10+ years of clinical use, creating genuine knowledge gaps. TVB can model network-level plasticity effects but requires extensions for long-term adaptation dynamics. Understanding responder biomarkers would directly improve patient selection.

**Source papers:** [PMID:6](https://pubmed.ncbi.nlm.nih.gov/6/) · [PMID:7](https://pubmed.ncbi.nlm.nih.gov/7/) · [PMID:70](https://pubmed.ncbi.nlm.nih.gov/70/) · [PMID:90](https://pubmed.ncbi.nlm.nih.gov/90/) · [PMID:182](https://pubmed.ncbi.nlm.nih.gov/182/)

_✅ TVB-addressable · 6 source ideas_

---

## #10 — Seizure forecasting across temporal scales
**Score: 21/30** &nbsp;|&nbsp; Novelty: 6/10 &nbsp;·&nbsp; TVB Feasibility: 7/10 &nbsp;·&nbsp; Clinical Impact: 8/10

**Summary:** Developing continuous forecasting systems predicting seizure risk at short-term, circadian, and multidien timescales using biomarkers, machine learning, and temporal pattern analysis.

**Representative idea:** _Multi-scale seizure forecasting systems predicting risk across short-term, circadian, and multidien timescales_

**Key opportunity:** Develop TVB-based state-space models that integrate ultradian rhythms for personalized seizure probability estimation.

**Rationale:** Multiscale forecasting integrating circadian/multidien cycles is gaining traction but not yet clinically deployed. TVB can incorporate slow fluctuations and bifurcation proximity metrics for proactive intervention timing. High patient quality-of-life impact if reliable forecasting achieved.

**Source papers:** [PMID:4](https://pubmed.ncbi.nlm.nih.gov/4/) · [PMID:21](https://pubmed.ncbi.nlm.nih.gov/21/) · [PMID:45](https://pubmed.ncbi.nlm.nih.gov/45/) · [PMID:151](https://pubmed.ncbi.nlm.nih.gov/151/) · [PMID:283](https://pubmed.ncbi.nlm.nih.gov/283/)

_✅ TVB-addressable · 5 source ideas_

---

## #11 — Cortico-cortical evoked potentials (CCEP) for seizure zone mapping and targeting
**Score: 21/30** &nbsp;|&nbsp; Novelty: 6/10 &nbsp;·&nbsp; TVB Feasibility: 8/10 &nbsp;·&nbsp; Clinical Impact: 7/10

**Summary:** Using CCEP-derived transfer functions, in-degree connectivity, and amplitude responses to predict optimal stimulation parameters, map seizure zones, and improve RNS electrode placement.

**Representative idea:** _CCEP transfer functions and in-degree connectivity for predicting effective stimulation parameters and seizure zone mapping_

**Key opportunity:** Validate TVB model predictions of stimulation propagation against empirical CCEP responses for model refinement.

**Rationale:** CCEP network mapping is established but systematic integration with computational models for target optimization is underexplored. TVB effective connectivity estimation aligns well with CCEP-derived directional measures. Clinical utility is moderate given CCEP already used clinically, but optimization potential exists.

**Source papers:** [PMID:44](https://pubmed.ncbi.nlm.nih.gov/44/) · [PMID:49](https://pubmed.ncbi.nlm.nih.gov/49/) · [PMID:72](https://pubmed.ncbi.nlm.nih.gov/72/) · [PMID:137](https://pubmed.ncbi.nlm.nih.gov/137/) · [PMID:170](https://pubmed.ncbi.nlm.nih.gov/170/)

_✅ TVB-addressable · 5 source ideas_

---

## #12 — Bifurcation-based seizure dynamics and control
**Score: 21/30** &nbsp;|&nbsp; Novelty: 5/10 &nbsp;·&nbsp; TVB Feasibility: 10/10 &nbsp;·&nbsp; Clinical Impact: 6/10

**Summary:** Using bifurcation analysis and Epileptor model extensions to characterize seizure mechanisms, design parameter modifications, and develop closed-loop control strategies targeting thalamocortical dynamics.

**Representative idea:** _Bifurcation analysis and Epileptor model for seizure classification and personalized control strategy design_

**Key opportunity:** Develop patient-specific bifurcation parameter estimation from clinical EEG to enable model-predictive stimulation control.

**Rationale:** Epileptor and bifurcation taxonomy are well-established within TVB framework, limiting novelty despite elegant theory. TVB is the primary platform for this approach with mature implementation. Clinical translation gap remains in mapping abstract bifurcation parameters to actionable interventions.

**Source papers:** [PMID:40](https://pubmed.ncbi.nlm.nih.gov/40/) · [PMID:71](https://pubmed.ncbi.nlm.nih.gov/71/) · [PMID:92](https://pubmed.ncbi.nlm.nih.gov/92/) · [PMID:93](https://pubmed.ncbi.nlm.nih.gov/93/) · [PMID:191](https://pubmed.ncbi.nlm.nih.gov/191/)

_✅ TVB-addressable · 6 source ideas_

---

## #13 — Structural connectome-based seizure propagation modeling
**Score: 21/30** &nbsp;|&nbsp; Novelty: 5/10 &nbsp;·&nbsp; TVB Feasibility: 9/10 &nbsp;·&nbsp; Clinical Impact: 7/10

**Summary:** Incorporating structural connectivity heterogeneity, regional excitability variations, and white matter organization into random walk and network models of seizure spread prediction.

**Representative idea:** _Structural connectome-based models incorporating excitability heterogeneity to predict patient-specific seizure propagation_

**Key opportunity:** Validate patient-specific excitability parameter estimation against SEEG recordings to close the gap between model predictions and observed propagation patterns.

**Rationale:** Connectome-based propagation modeling is an active field with established methods, though incorporating regional excitability heterogeneity adds incremental novelty. TVB is purpose-built for this application with existing structural connectivity pipelines. Direct clinical utility for surgical planning if validated prospectively.

**Source papers:** [PMID:61](https://pubmed.ncbi.nlm.nih.gov/61/) · [PMID:145](https://pubmed.ncbi.nlm.nih.gov/145/) · [PMID:157](https://pubmed.ncbi.nlm.nih.gov/157/) · [PMID:271](https://pubmed.ncbi.nlm.nih.gov/271/) · [PMID:272](https://pubmed.ncbi.nlm.nih.gov/272/)

_✅ TVB-addressable · 5 source ideas_

---

## #14 — Deep brain stimulation (DBS) mechanisms and network effects
**Score: 21/30** &nbsp;|&nbsp; Novelty: 6/10 &nbsp;·&nbsp; TVB Feasibility: 7/10 &nbsp;·&nbsp; Clinical Impact: 8/10

**Summary:** Investigating how DBS suppresses seizures through potassium accumulation, axonal depolarization block, desynchronization, and whole-brain network modulation with parameter optimization.

**Representative idea:** _DBS mechanisms including potassium dynamics, axonal block, and network desynchronization for seizure suppression_

**Key opportunity:** Develop multi-scale TVB extensions linking local ionic/axonal mechanisms to whole-brain network desynchronization to optimize target selection.

**Rationale:** DBS mechanisms remain incompletely understood despite clinical use; potassium dynamics and axonal block are moderately novel angles. TVB can model network-level effects but requires coupling with detailed biophysical models for ionic dynamics. High clinical relevance given growing DBS use in drug-resistant epilepsy.

**Source papers:** [PMID:8](https://pubmed.ncbi.nlm.nih.gov/8/) · [PMID:40](https://pubmed.ncbi.nlm.nih.gov/40/) · [PMID:134](https://pubmed.ncbi.nlm.nih.gov/134/) · [PMID:194](https://pubmed.ncbi.nlm.nih.gov/194/) · [PMID:259](https://pubmed.ncbi.nlm.nih.gov/259/)

_✅ TVB-addressable · 6 source ideas_

---

## #15 — Functional network reorganization during electrical stimulation
**Score: 21/30** &nbsp;|&nbsp; Novelty: 7/10 &nbsp;·&nbsp; TVB Feasibility: 8/10 &nbsp;·&nbsp; Clinical Impact: 6/10

**Summary:** Examining stimulus-induced shifts from integrated to distributed connectivity patterns and relating network topology evolution to seizure zone properties and propagation mechanisms.

**Representative idea:** _Functional network reorganization during stimulation relates to seizure zone properties and propagation mechanisms_

**Key opportunity:** Establish quantitative TVB-derived metrics of stimulation-induced network reorganization that predict seizure freedom.

**Rationale:** Real-time network topology shifts during stimulation are underexplored compared to baseline functional connectivity; integrated-to-distributed transition is a fresh angle. TVB excels at modeling perturbation-induced network reorganization. Clinical translation requires linking topology changes to therapeutic outcomes.

**Source papers:** [PMID:215](https://pubmed.ncbi.nlm.nih.gov/215/) · [PMID:216](https://pubmed.ncbi.nlm.nih.gov/216/) · [PMID:217](https://pubmed.ncbi.nlm.nih.gov/217/) · [PMID:252](https://pubmed.ncbi.nlm.nih.gov/252/)

_✅ TVB-addressable · 4 source ideas_

---

## #16 — Clinical validation and prospective outcome studies
**Score: 21/30** &nbsp;|&nbsp; Novelty: 4/10 &nbsp;·&nbsp; TVB Feasibility: 8/10 &nbsp;·&nbsp; Clinical Impact: 9/10

**Summary:** Implementing multi-center prospective studies to validate computational models, biomarkers, seizure detection algorithms, and treatment planning approaches in real-world epilepsy populations.

**Representative idea:** _Multi-center prospective clinical validation of computational models and biomarkers in drug-resistant epilepsy_

**Key opportunity:** Launch a consortium trial comparing TVB-guided surgical planning against standard care with seizure-freedom as primary endpoint.

**Rationale:** Prospective validation studies are methodologically standard but critically needed; the novelty lies in validating computational predictions specifically. TVB is ideally positioned as the model requiring validation, making this highly feasible. Clinical impact is very high as successful validation would transform computational tools into clinical decision aids.

**Source papers:** [PMID:30](https://pubmed.ncbi.nlm.nih.gov/30/) · [PMID:55](https://pubmed.ncbi.nlm.nih.gov/55/) · [PMID:62](https://pubmed.ncbi.nlm.nih.gov/62/) · [PMID:81](https://pubmed.ncbi.nlm.nih.gov/81/) · [PMID:135](https://pubmed.ncbi.nlm.nih.gov/135/)

_✅ TVB-addressable · 6 source ideas_

---

## #17 — Behavioral and emotional signal integration into neuromodulation
**Score: 21/30** &nbsp;|&nbsp; Novelty: 7/10 &nbsp;·&nbsp; TVB Feasibility: 7/10 &nbsp;·&nbsp; Clinical Impact: 7/10

**Summary:** Incorporating patient behavioral, emotional, and cognitive signals into adaptive neuromodulation systems using connectomics-informed decoding for multi-domain monitoring.

**Representative idea:** _Connectomics-informed neural decoding of emotional and cognitive states for adaptive neuromodulation_

**Key opportunity:** Develop TVB modules that simulate how mood/cognitive states alter seizure network dynamics to personalize stimulation timing.

**Rationale:** Multi-domain behavioral/emotional integration into adaptive stimulation is underexplored beyond seizure detection; most systems ignore cognitive/affective state. TVB's connectome-based framework naturally supports modeling limbic-cortical interactions and state-dependent dynamics. High clinical relevance given psychiatric comorbidities in epilepsy.

**Source papers:** [PMID:13](https://pubmed.ncbi.nlm.nih.gov/13/) · [PMID:24](https://pubmed.ncbi.nlm.nih.gov/24/) · [PMID:78](https://pubmed.ncbi.nlm.nih.gov/78/) · [PMID:111](https://pubmed.ncbi.nlm.nih.gov/111/)

_✅ TVB-addressable · 4 source ideas_

---

## #18 — Multimodal neuroimaging integration for seizure zone localization
**Score: 20/30** &nbsp;|&nbsp; Novelty: 4/10 &nbsp;·&nbsp; TVB Feasibility: 8/10 &nbsp;·&nbsp; Clinical Impact: 8/10

**Summary:** Combining structural MRI, diffusion tractography, functional connectivity, PET, sodium MRI, and electrophysiology within computational frameworks to improve EZ localization and surgical outcome prediction.

**Representative idea:** _Multimodal imaging integration (structural, diffusion, functional, PET, sodium MRI) within computational frameworks for EZ localization_

**Key opportunity:** Develop automated TVB pipelines that weight multimodal inputs optimally based on availability and quality to maximize EZ localization accuracy.

**Rationale:** Multimodal integration is actively pursued by many groups; adding sodium MRI and PET provides incremental advance. TVB is designed for multimodal integration and can naturally incorporate these as regional parameters. High clinical impact through improved concordance and reduced invasive monitoring.

**Source papers:** [PMID:47](https://pubmed.ncbi.nlm.nih.gov/47/) · [PMID:65](https://pubmed.ncbi.nlm.nih.gov/65/) · [PMID:153](https://pubmed.ncbi.nlm.nih.gov/153/) · [PMID:167](https://pubmed.ncbi.nlm.nih.gov/167/) · [PMID:175](https://pubmed.ncbi.nlm.nih.gov/175/)

_✅ TVB-addressable · 5 source ideas_

---

## #19 — AI-assisted clinical decision support for surgical planning
**Score: 19/30** &nbsp;|&nbsp; Novelty: 4/10 &nbsp;·&nbsp; TVB Feasibility: 6/10 &nbsp;·&nbsp; Clinical Impact: 9/10

**Summary:** Developing AI tools that integrate neuroimaging, electrophysiology, and computational models to guide electrode placement, predict surgical outcomes, and optimize patient selection.

**Representative idea:** _AI clinical decision support integrating neuroimaging and computational models for surgical planning and outcome prediction_

**Key opportunity:** Prospective multi-center validation trial comparing AI-TVB recommendations against standard presurgical evaluation outcomes.

**Rationale:** AI for surgical planning is heavily pursued by multiple groups with existing commercial efforts; novelty is limited. TVB integration is feasible but requires robust pipelines for multi-modal data fusion and model outputs as AI features. Extremely high clinical impact if validated—could directly improve surgical decision-making.

**Source papers:** [PMID:26](https://pubmed.ncbi.nlm.nih.gov/26/) · [PMID:30](https://pubmed.ncbi.nlm.nih.gov/30/) · [PMID:53](https://pubmed.ncbi.nlm.nih.gov/53/) · [PMID:62](https://pubmed.ncbi.nlm.nih.gov/62/) · [PMID:156](https://pubmed.ncbi.nlm.nih.gov/156/)

_✅ TVB-addressable · 6 source ideas_

---

## #20 — Post-ictal diffusion imaging for seizure propagation mapping
**Score: 19/30** &nbsp;|&nbsp; Novelty: 8/10 &nbsp;·&nbsp; TVB Feasibility: 4/10 &nbsp;·&nbsp; Clinical Impact: 7/10

**Summary:** Expanding post-ictal spectral DTI (pspiDTI) to detect water diffusivity abnormalities in various seizure types and correlating findings with seizure propagation networks for improved surgical planning.

**Representative idea:** _Post-ictal spectral DTI detecting transient white matter diffusivity abnormalities for seizure propagation mapping_

**Key opportunity:** Correlate pspiDTI abnormalities with TVB-predicted propagation pathways to create hybrid structural-functional maps for surgical planning.

**Rationale:** Post-ictal spectral DTI is a novel imaging approach with limited prior work; expanding to various seizure types is genuinely underexplored. TVB does not directly process DTI dynamics but could incorporate findings as propagation constraints. Clinical utility is clear if post-ictal imaging becomes logistically feasible.

**Source papers:** [PMID:193](https://pubmed.ncbi.nlm.nih.gov/193/) · [PMID:236](https://pubmed.ncbi.nlm.nih.gov/236/) · [PMID:237](https://pubmed.ncbi.nlm.nih.gov/237/) · [PMID:238](https://pubmed.ncbi.nlm.nih.gov/238/) · [PMID:239](https://pubmed.ncbi.nlm.nih.gov/239/)

_⚙️ Needs extension · 6 source ideas_

---
