# 3D Indoor Scene Synthesis — Paper Survey

> 기초 논문 ⭐ 표시
>
> 참고: [Awesome-Indoor-Scene-Synthesis](https://github.com/YandanYang/Awesome-Indoor-Scene-Synthesis) · [Awesome-3D-Scene-Generation](https://github.com/hzxie/Awesome-3D-Scene-Generation)

## 목차

1. [Representation & Understanding](#1-representation--understanding)
2. [Layout Generation (Furniture Arrangement)](#2-layout-generation-furniture-arrangement)
3. [Scene Generation with Language / Multimodal Input](#3-scene-generation-with-language--multimodal-input)
4. [Full 3D Scene Generation (Geometry + Texture)](#4-full-3d-scene-generation-geometry--texture)
5. [Asset & Texture Synthesis](#5-asset--texture-synthesis)
6. [Applications](#6-applications)
7. [Datasets & Evaluation](#7-datasets--evaluation)

---

## 1. Representation & Understanding

### 1.1 Scene Reconstruction (NeRF, 3DGS)

#### ⭐ NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis
- **저자**: Ben Mildenhall et al. | **ECCV 2020** | 인용: ~17,000+
- **논문**: https://arxiv.org/abs/2003.08934
- **프로젝트**: https://www.matthewtancik.com/nerf | [GitHub](https://github.com/bmild/nerf)
- **요약**: 연속적인 5D 함수(위치+방향)를 MLP로 학습해 새로운 시점의 이미지를 렌더링. 3D scene representation의 패러다임을 바꾼 논문.

#### ⭐ 3D Gaussian Splatting for Real-Time Radiance Field Rendering
- **저자**: Bernhard Kerbl et al. | **SIGGRAPH 2023** | 인용: ~3,000+
- **논문**: https://arxiv.org/abs/2308.04079
- **프로젝트**: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/ | [GitHub](https://github.com/graphdeco-inria/gaussian-splatting)
- **요약**: 3D 가우시안을 명시적 표현으로 사용해 실시간 렌더링 가능. NeRF의 느린 추론 문제를 해결.

#### Instant Neural Graphics Primitives with a Multiresolution Hash Encoding
- **저자**: Thomas Müller et al. | **SIGGRAPH 2022** | 인용: ~3,000+
- **논문**: https://arxiv.org/abs/2201.05989
- **프로젝트**: https://nvlabs.github.io/instant-ngp/ | [GitHub](https://github.com/NVlabs/instant-ngp)
- **요약**: 해시 인코딩 기반 즉각적 NeRF 학습으로 훈련 속도를 수십 배 향상.

---

### 1.2 Scene Graph / Relation Modeling

#### ⭐ SceneGraphNet: Neural Message Passing for 3D Indoor Scene Augmentation
- **저자**: Yang Zhou et al. | **ICCV 2019**
- **논문**: https://arxiv.org/abs/1907.11308
- **프로젝트**: https://people.umass.edu/~yangzhou/scenegraphnet/ | [GitHub](https://github.com/yzhou359/3DIndoor-SceneGraphNet)
- **요약**: 실내 장면을 그래프로 모델링하고, 노드(오브젝트)와 엣지(관계)를 통해 새로운 오브젝트 배치를 예측.

#### 3D Scene Graph: A Structure for Unified Semantics, 3D Space, and Camera
- **저자**: Iro Armeni et al. | **ICCV 2019**
- **논문**: https://arxiv.org/abs/1910.02527
- **프로젝트**: http://3dscenegraph.stanford.edu | [GitHub](https://github.com/StanfordVL/3DSceneGraph)
- **요약**: 실내 공간을 계층적 그래프(건물→룸→오브젝트)로 표현. 통합 semantic 표현 제공.

#### Holistic 3D Scene Understanding from a Single Image with Implicit Representation
- **저자**: Cheng Zhang et al. | **CVPR 2021**
- **논문**: https://arxiv.org/abs/2103.06422
- **프로젝트**: https://chengzhag.github.io/publication/im3d/ | [GitHub](https://github.com/chengzhag/Implicit3DUnderstanding)
- **요약**: 단일 이미지에서 레이아웃·오브젝트·관계를 동시에 추론하는 implicit representation 기반 holistic 3D 이해.

---

### 1.3 Layout Estimation from Images

#### ⭐ Total3DUnderstanding: Joint Layout, Object Pose and Mesh Reconstruction for Indoor Scenes from a Single Image
- **저자**: Yinyu Nie et al. | **CVPR 2020** | 인용: ~400+
- **논문**: https://arxiv.org/abs/2002.12212
- **프로젝트**: https://yinyunie.github.io/Total3D/ | [GitHub](https://github.com/GAP-LAB-CUHK-SZ/Total3DUnderstanding)
- **요약**: 단일 RGB 이미지에서 방 레이아웃, 오브젝트 포즈, 메시를 동시에 복원. 3D indoor understanding의 end-to-end 대표 연구.

---

## 2. Layout Generation (Furniture Arrangement)

### 2.1 Optimization-based / CNN-based (초기 방법론)

#### ⭐ Configurable 3D Scene Synthesis and 2D Image Rendering with Per-pixel Ground Truth Using Stochastic Grammars
- **저자**: Siyuan Qi et al. | **IJCV 2018** | 인용: ~200+
- **논문**: https://arxiv.org/abs/1704.00112
- **프로젝트**: https://yzhu.io/publication/scenesynthesis2018ijcv/
- **요약**: 확률적 문법과 MCMC를 결합해 물리적으로 타당한 실내 장면을 생성. 규칙 기반 장면 합성의 대표 연구.

#### Human-centric Indoor Scene Synthesis Using Stochastic Grammar
- **저자**: Siyuan Qi et al. | **CVPR 2018** | 인용: ~200+
- **논문**: https://arxiv.org/abs/1808.08473
- **프로젝트**: [GitHub](https://github.com/SiyuanQi/human-centric-scene-synthesis)
- **요약**: 인간 활동 패턴을 고려한 확률 문법 기반 실내 장면 합성.

#### ⭐ Deep Convolutional Priors for Indoor Scene Synthesis
- **저자**: Kai Wang et al. | **SIGGRAPH 2018** | 인용: **261**
- **논문**: https://doi.org/10.1145/3197517.3201362
- **프로젝트**: [GitHub](https://github.com/brownvc/deep-synth)
- **요약**: CNN을 활용해 실내 장면의 오브젝트 배치를 순차적으로 생성. 딥러닝 기반 indoor scene synthesis의 선구적 연구.

#### ⭐ Fast and Flexible Indoor Scene Synthesis via Deep Convolutional Generative Models
- **저자**: Daniel Ritchie et al. | **CVPR 2019** | 인용: **178**
- **논문**: https://arxiv.org/abs/1811.12463
- **프로젝트**: [GitHub](https://github.com/brownvc/fast-synth)
- **요약**: Deep CNN 생성 모델로 빠르고 다양한 실내 장면 합성. Transformer/Diffusion 기반 연구들의 직접적 선구.

---

### 2.2 Autoregressive Models (Transformer)

#### ⭐ ATISS: Autoregressive Transformers for Indoor Scene Synthesis
- **저자**: Despoina Paschalidou et al. | **NeurIPS 2021** | 인용: **171**
- **논문**: https://arxiv.org/abs/2110.03675
- **프로젝트**: https://nv-tlabs.github.io/ATISS/ | [GitHub](https://github.com/nv-tlabs/ATISS)
- **요약**: Transformer의 자기회귀 생성 능력을 실내 장면 오브젝트 배치에 적용한 대표 연구. 이후 diffusion 기반 방법론의 비교 baseline으로 광범위하게 사용됨.

#### ⭐ SceneFormer: Indoor Scene Generation with Transformers
- **저자**: Xinpeng Wang et al. | **3DV 2021** | 인용: **164**
- **논문**: https://arxiv.org/abs/2012.09793
- **프로젝트**: https://xinpeng-wang.github.io/sceneformer/ | [GitHub](https://github.com/cy94/sceneformer)
- **요약**: Transformer 기반 순차적 오브젝트 배치 생성. ATISS와 함께 자기회귀 장면 합성의 양대 기반.

#### Scene Synthesis via Uncertainty-Driven Attribute Synchronization (Sync2Gen)
- **저자**: Haitao Yang et al. | **ICCV 2021**
- **논문**: https://arxiv.org/abs/2108.13499
- **프로젝트**: [GitHub](https://github.com/yanghtr/Sync2Gen)
- **요약**: 불확실성 기반 속성 동기화로 오브젝트 간 일관성 있는 실내 장면 합성.

#### CLIP-Layout: Style-Consistent Indoor Scene Synthesis with Semantic Furniture Embedding
- **저자**: Jingyu Liu et al. | **arXiv 2023** | 인용: 19
- **논문**: https://arxiv.org/abs/2303.03565
- **요약**: CLIP 임베딩으로 스타일 일관성을 유지하면서 자기회귀적 가구 배치 생성.

---

### 2.3 Diffusion Models

#### ⭐ DiffuScene: Denoising Diffusion Models for Generative Indoor Scene Synthesis
- **저자**: Jiapeng Tang et al. | **CVPR 2024** | 인용: **119**
- **논문**: https://arxiv.org/abs/2303.14207
- **프로젝트**: https://tangjiapeng.github.io/projects/DiffuScene/ | [GitHub](https://github.com/tangjiapeng/DiffuScene)
- **요약**: 확산 모델(DDPM)을 실내 장면 레이아웃 생성에 최초로 적용한 핵심 연구. 이후 diffusion 기반 scene synthesis의 표준 baseline.

#### InstructScene: Instruction-Driven 3D Indoor Scene Synthesis with Semantic Graph Prior
- **저자**: Chenguo Lin, Yadong Mu | **ICLR 2024 (Spotlight)** | 인용: **80**
- **논문**: https://arxiv.org/abs/2402.04717
- **프로젝트**: https://chenguolin.github.io/projects/InstructScene | [GitHub](https://github.com/chenguolin/InstructScene)
- **요약**: 텍스트 instruction과 semantic graph prior를 결합한 diffusion 기반 장면 생성. (Graph 기반 요소도 핵심)

#### Mixed Diffusion for 3D Indoor Scene Synthesis (MiDiffusion)
- **저자**: Siyi Hu et al. | **arXiv 2024** | 인용: 15
- **논문**: https://arxiv.org/abs/2405.21066
- **프로젝트**: [GitHub](https://github.com/MIT-SPARK/MiDiffusion)
- **요약**: 연속형·이산형 속성을 통합 처리하는 혼합 확산 모델로 다양한 오브젝트 배치 생성.

#### DiffInDScene: Diffusion-Based High-Quality 3D Indoor Scene Generation
- **저자**: (2023) | **CVPR 2023** | 인용: 20
- **논문**: https://arxiv.org/abs/2306.00519
- **프로젝트**: [GitHub](https://github.com/AkiraHero/diffindscene)
- **요약**: 고품질 3D indoor scene을 위한 diffusion 기반 파이프라인.

#### SemLayoutDiff: Semantic Layout Generation with Diffusion Model for Indoor Scene Synthesis
- **저자**: Xiaohao Sun et al. | **arXiv 2025** | 인용: 3
- **논문**: https://arxiv.org/abs/2508.18597
- **프로젝트**: https://3dlg-hcvc.github.io/SemLayoutDiff/
- **요약**: 의미론적 레이아웃을 먼저 생성 후 세부 합성하는 계층적 diffusion 접근.

---

### 2.4 GAN / VAE-based

#### RoomDesigner: Encoding Anchor-latents for Style-consistent and Shape-compatible Indoor Scene Generation
- **저자**: (2024) | **3DV 2024**
- **논문**: https://arxiv.org/abs/2310.10027
- **프로젝트**: [GitHub](https://github.com/zhao-yiqun/RoomDesigner)
- **요약**: VQ-VAE + 자기회귀 transformer 하이브리드로 스타일 일관성과 형태 호환성을 갖춘 실내 장면 생성.

#### Learning Graph Variational Autoencoders with Constraints and Structured Priors for Conditional Indoor 3D Scene Generation
- **저자**: Chattopadhyay et al. | **WACV 2023** | 인용: 11
- **논문**: https://openaccess.thecvf.com/content/WACV2023/papers/Chattopadhyay_Learning_Graph_Variational_Autoencoders_With_Constraints_and_Structured_Priors_for_WACV_2023_paper.pdf
- **요약**: 구조적 prior와 제약 조건을 갖춘 그래프 VAE 기반 실내 3D 장면 생성.

---

### 2.5 Graph-based (Scene Graph)

#### ⭐ CommonScenes: Generating Commonsense 3D Indoor Scenes with Scene Graph Diffusion
- **저자**: Guangyao Zhai et al. | **NeurIPS 2023** | 인용: ~60+
- **논문**: https://arxiv.org/abs/2305.16283
- **프로젝트**: https://sites.google.com/view/commonscenes | [GitHub](https://github.com/ymxlzgy/commonscenes)
- **요약**: 상식적 공간 관계를 장면 그래프로 인코딩하고 diffusion으로 3D 장면 생성. Scene graph + diffusion 결합의 선구적 연구.

#### EchoScene: Indoor Scene Generation via Information Echo over Scene Graph Diffusion
- **저자**: (2024) | **ECCV 2024**
- **논문**: https://arxiv.org/abs/2405.00915
- **프로젝트**: [GitHub](https://github.com/ymxlzgy/echoscene)
- **요약**: 씬 그래프의 정보 전파(echo)를 활용한 diffusion 기반 장면 생성.

#### FreeScene: Mixed Graph Diffusion for 3D Scene Synthesis from Free Prompts
- **저자**: (2025) | **CVPR 2025**
- **논문**: https://arxiv.org/abs/2506.02781
- **요약**: 자유 형식 프롬프트에서 혼합 그래프 diffusion으로 3D 장면 합성.

#### MMGDreamer: Mixed-Modality Graph for Geometry-Controllable 3D Indoor Scene Generation
- **저자**: (2025) | **AAAI 2025** | 인용: 20
- **논문**: https://arxiv.org/abs/2502.05874
- **프로젝트**: https://yangzhifeio.github.io/project/MMGDreamer | [GitHub](https://github.com/yangzhifeio/MMGDreamer)
- **요약**: 텍스트·이미지 혼합 모달리티 그래프로 기하학적으로 제어 가능한 장면 생성.

#### 3D Scene Diffusion Guidance using Scene Graphs
- **저자**: (2023)
- **논문**: https://arxiv.org/abs/2308.04468
- **프로젝트**: [GitHub](https://github.com/hamnaanaa/3D-Scene-Diffusion-Guidance-using-Scene-Graphs)
- **요약**: 씬 그래프를 guidance로 사용하는 3D 장면 diffusion 생성.

---

## 3. Scene Generation with Language / Multimodal Input

### 3.1 Text-to-Scene (LLM-based)

#### ⭐ Holodeck: Language Guided Generation of 3D Embodied AI Environments
- **저자**: Yue Yang et al. | **CVPR 2024** | 인용: ~100+
- **논문**: https://arxiv.org/abs/2312.09067
- **프로젝트**: https://yueyang1996.github.io/holodeck/ | [GitHub](https://github.com/allenai/Holodeck)
- **요약**: GPT-4를 활용해 텍스트 설명에서 Embodied AI용 3D 환경을 자동 생성. LLM 기반 장면 생성의 대표작.

#### Open-Universe Indoor Scene Generation using LLM Program Synthesis and Uncurated Object Databases
- **저자**: (2024) | 인용: 42
- **논문**: https://arxiv.org/abs/2403.09675
- **요약**: LLM 프로그램 합성과 비정제 오브젝트 DB를 결합해 오픈 유니버스 실내 장면 생성.

#### SceneTeller: Language-to-3D Scene Generation
- **저자**: Basak Melis Öcal et al. | **ECCV 2024** | 인용: 44
- **논문**: https://arxiv.org/abs/2407.20727
- **프로젝트**: https://sceneteller.github.io/ | [GitHub](https://github.com/sceneteller/SceneTeller)
- **요약**: 텍스트를 3D 장면으로 직접 변환하는 언어 기반 장면 생성 파이프라인.

#### LLplace: The 3D Indoor Scene Layout Generation and Editing via Large Language Model
- **저자**: (2024) | 인용: 22
- **논문**: https://arxiv.org/abs/2406.03866
- **요약**: LLM을 이용해 3D 실내 장면 레이아웃 생성 및 편집.

#### 3D-GPT: Procedural 3D Modeling with Large Language Models
- **저자**: (2023)
- **논문**: https://arxiv.org/abs/2310.12945
- **프로젝트**: https://chuny1.github.io/3DGPT/3dgpt.html | [GitHub](https://github.com/Chuny1/3DGPT)
- **요약**: LLM을 이용한 절차적 3D 모델링. 텍스트→Python 코드→3D 장면 파이프라인.

#### SceneWeaver: All-in-One 3D Scene Synthesis with an Extensible and Self-Reflective Agent
- **저자**: (2025) | **NeurIPS 2025**
- **논문**: https://arxiv.org/abs/2509.20414
- **프로젝트**: https://scene-weaver.github.io/ | [GitHub](https://github.com/Scene-Weaver/SceneWeaver)
- **요약**: 자기 반성적(self-reflective) 에이전트로 확장 가능한 올인원 3D 장면 합성.

#### ArtiScene: Language-Driven Artistic 3D Scene Generation Through Image Intermediary
- **저자**: (2025) | **CVPR 2025**
- **논문**: https://arxiv.org/abs/2506.00742
- **프로젝트**: [GitHub](https://github.com/NVlabs/ArtiScene)
- **요약**: 이미지를 중간 매개체로 사용하는 언어 기반 아티스틱 3D 장면 생성.

#### Scenethesis: A Language and Vision Agentic Framework for 3D Scene Generation
- **저자**: Lu Ling et al. | **arXiv 2025** | 인용: 34
- **논문**: https://arxiv.org/abs/2505.02836
- **요약**: 언어와 비전을 결합한 에이전트 프레임워크로 3D 장면 생성.

#### Global-Local Tree Search in VLMs for 3D Indoor Scene Generation
- **저자**: (2025) | **CVPR 2025**
- **논문**: https://arxiv.org/abs/2503.18476
- **프로젝트**: [GitHub](https://github.com/dw-dengwei/TreeSearchGen)
- **요약**: VLM의 전역-지역 트리 탐색을 활용한 3D 실내 장면 생성.

#### Hierarchically-Structured Open-Vocabulary Indoor Scene Synthesis with Pre-trained Large Language Model
- **저자**: Weilin Sun et al. | **AAAI 2025** | 인용: 6
- **논문**: https://arxiv.org/abs/2502.10675
- **요약**: 사전학습 LLM의 계층적 구조를 활용한 오픈 보캐뷸러리 실내 장면 합성.

#### ReSpace: Text-Driven Autoregressive 3D Indoor Scene Synthesis and Editing
- **저자**: M. Bucher et al. | **arXiv 2025** | 인용: 2
- **논문**: https://arxiv.org/abs/2506.02459
- **요약**: 텍스트 기반 자기회귀 3D 실내 장면 합성 및 편집.

#### SceneSmith: Agentic Generation of Simulation-Ready Indoor Scenes
- **저자**: (2026)
- **논문**: https://arxiv.org/abs/2602.09153
- **프로젝트**: https://scenesmith.github.io/ | [GitHub](https://github.com/nepfaff/scenesmith)
- **요약**: 계층적 에이전트 프레임워크로 시뮬레이션 준비된 실내 장면을 NLP 프롬프트에서 생성.

#### WorldCraft: Photo-Realistic 3D World Creation and Customization via LLM Agents
- **저자**: (2025)
- **논문**: https://arxiv.org/abs/2502.15601
- **요약**: LLM 에이전트를 통한 포토리얼리스틱 3D 세계 창조 및 커스터마이징.

---

### 3.2 Image-conditioned Generation

#### Lay-A-Scene: Personalized 3D Object Arrangement Using Text-to-Image Priors
- **저자**: (2024) | 인용: ~20+
- **논문**: https://arxiv.org/abs/2406.00687
- **프로젝트**: https://lay-a-scene.github.io/
- **요약**: 텍스트→이미지 prior를 활용한 개인화된 3D 오브젝트 배치.

#### MIDI: Multi-Instance Diffusion for Single Image to 3D Scene Generation
- **저자**: (2024) | **CVPR 2025**
- **논문**: https://arxiv.org/abs/2412.03558
- **프로젝트**: https://huanngzh.github.io/MIDI-Page/ | [GitHub](https://github.com/VAST-AI-Research/MIDI-3D)
- **요약**: 단일 이미지에서 다중 인스턴스 diffusion을 통해 3D 장면 생성.

---

### 3.3 Human Motion-conditioned

#### ⭐ Pose2Room: Understanding 3D Scenes from Human Activities
- **저자**: (2022) | **ECCV 2022**
- **논문**: https://arxiv.org/abs/2112.03030
- **프로젝트**: https://yinyunie.github.io/pose2room-page/ | [GitHub](https://github.com/yinyunie/Pose2Room)
- **요약**: 인간 자세 시퀀스를 조건으로 3D 실내 장면을 이해·생성. 인간 활동 기반 장면 합성의 초기 핵심 연구.

#### MIME: Human-Aware 3D Scene Generation
- **저자**: (2023) | **CVPR 2023**
- **논문**: https://arxiv.org/abs/2212.04360
- **프로젝트**: https://mime.is.tue.mpg.de/ | [GitHub](https://github.com/yhw-yhw/MIME)
- **요약**: 인간 동작 인식 기반 3D 장면 생성. 인간과 환경의 상호작용을 고려한 배치.

#### SUMMON: Scene Synthesis from Human Motion
- **저자**: (2022) | **SIGGRAPH Asia 2022**
- **논문**: https://arxiv.org/abs/2301.01424
- **프로젝트**: https://lijiaman.github.io/projects/summon/ | [GitHub](https://github.com/onestarYX/summon)
- **요약**: 인간 모션 데이터에서 직접 씬을 합성하는 방법론.

#### Human-Aware 3D Scene Generation with Spatially-constrained Diffusion Models (SHADE)
- **저자**: (2024)
- **논문**: https://arxiv.org/abs/2406.18159
- **요약**: 공간 제약 diffusion 모델로 인간 인식 3D 장면 생성.

#### Rearrange Indoor Scenes for Human-Robot Co-Activity
- **저자**: (2023) | **ICRA 2023**
- **논문**: https://arxiv.org/abs/2303.05676
- **프로젝트**: https://sites.google.com/view/coactivity | [GitHub](https://github.com/Rayckey/scene_coactivity)
- **요약**: 인간-로봇 협업 활동을 위한 실내 장면 재배치.

---

## 4. Full 3D Scene Generation (Geometry + Texture)

### 4.1 Procedural / Rule-based

#### ⭐ Infinigen Indoors: Photorealistic Indoor Scenes using Procedural Generation
- **저자**: (2024) | **CVPR 2024** | 인용: **102**
- **논문**: https://arxiv.org/abs/2406.11824
- **프로젝트**: https://infinigen.org | [GitHub](https://github.com/princeton-vl/infinigen)
- **요약**: 절차적 생성으로 포토리얼리스틱한 실내 장면 데이터를 무한 생성. Blender 기반 완전 자동화 파이프라인.

#### ⭐ ProcTHOR: Large-Scale Embodied AI Using Procedural Generation
- **저자**: (2022) | **NeurIPS 2022 (Outstanding Paper)** | 인용: ~300+
- **논문**: https://arxiv.org/abs/2206.06994
- **프로젝트**: https://procthor.allenai.org | [GitHub](https://github.com/allenai/procthor)
- **요약**: 절차적 생성으로 대규모 AI2-THOR 기반 실내 환경 생성. Embodied AI 훈련용 데이터 생성의 표준.

---

### 4.2 Neural Compositional (Object Assembly)

#### GALA3D: Towards Text-to-3D Complex Scene Generation via Layout-guided Generative Gaussian Splatting
- **저자**: (2024) | **ICML 2024** | 인용: ~50+
- **논문**: https://arxiv.org/abs/2402.07207
- **프로젝트**: https://gala3d.github.io/ | [GitHub](https://github.com/VDIGPKU/GALA3D)
- **요약**: 레이아웃 가이드 생성 가우시안 스플래팅으로 텍스트에서 복잡한 3D 장면 생성.

#### SceneWiz3D: Towards Text-guided 3D Scene Composition
- **저자**: (2023) | **CVPR 2024** | 인용: ~80+
- **논문**: https://arxiv.org/abs/2312.08885
- **프로젝트**: https://zqh0253.github.io/SceneWiz3D/ | [GitHub](https://github.com/zqh0253/SceneWiz3D)
- **요약**: 텍스트 가이드로 여러 3D 오브젝트를 조합해 일관된 씬 구성.

#### GraphDreamer: Compositional 3D Scene Synthesis from Scene Graphs
- **저자**: (2023) | **CVPR 2024** | 인용: ~60+
- **논문**: https://arxiv.org/abs/2312.00093
- **프로젝트**: https://graphdreamer.github.io/ | [GitHub](https://github.com/GGGHSL/GraphDreamer)
- **요약**: 씬 그래프에서 compositional 3D 장면 합성.

#### Compositional 3D Scene Generation using Locally Conditioned Diffusion
- **저자**: (2023) | **3DV 2024** | 인용: ~30+
- **논문**: https://arxiv.org/abs/2303.12218
- **프로젝트**: https://ryanpo.com/comp3d/
- **요약**: 지역 조건화 diffusion으로 합성적 3D 장면 생성.

#### SceneFactor: Factored Latent 3D Diffusion for Controllable 3D Scene Generation
- **저자**: (2024) | **CVPR 2025**
- **논문**: https://arxiv.org/abs/2412.01801
- **프로젝트**: [GitHub](https://github.com/alexeybokhovkin/SceneFactor)
- **요약**: 분해된 latent diffusion으로 제어 가능한 3D 장면 생성.

---

### 4.3 End-to-End Neural (Mesh)

#### HouseCrafter: Lifting Floorplans to 3D Scenes with 2D Diffusion Models
- **저자**: (2025) | **ICCV 2025 (Highlight)**
- **논문**: https://arxiv.org/abs/2406.20077
- **프로젝트**: https://neu-vi.github.io/houseCrafter/ | [GitHub](https://github.com/neu-vi/houseCrafter)
- **요약**: 2D 평면도에서 2D diffusion을 활용해 완전한 3D 장면으로 변환.

#### MVRoom: Controllable 3D Indoor Scene Generation with Multi-View Diffusion Models
- **저자**: (2025)
- **논문**: https://arxiv.org/abs/2512.04248
- **요약**: 멀티뷰 diffusion으로 제어 가능한 3D 실내 장면 생성.

#### SceneNAT: Masked Generative Modeling for Language-Guided Indoor Scene Synthesis
- **저자**: (2026)
- **논문**: https://arxiv.org/abs/2601.07218
- **요약**: 언어 가이드 실내 장면 합성을 위한 masked generative modeling.

---

### 4.4 Using 2D Diffusion Prior

#### ARCHITECT: Generating Vivid and Interactive 3D Scenes with Hierarchical 2D Inpainting
- **저자**: (2024) | **NeurIPS 2024**
- **논문**: https://arxiv.org/abs/2411.09823
- **프로젝트**: https://wangyian-me.github.io/Architect/
- **요약**: 계층적 2D inpainting을 통해 생생하고 인터랙티브한 3D 장면 생성.

#### Disentangled 3D Scene Generation with Layout Learning
- **저자**: (2024) | **ICML 2024**
- **논문**: https://arxiv.org/abs/2402.16936
- **프로젝트**: https://dave.ml/layoutlearning/
- **요약**: 레이아웃 학습을 통한 disentangled 3D 장면 생성.

#### Learning 3D Scene Priors with 2D Supervision
- **저자**: (2023) | **CVPR 2023**
- **논문**: https://arxiv.org/abs/2211.14157
- **프로젝트**: https://yinyunie.github.io/sceneprior-page/ | [GitHub](https://github.com/yinyunie/ScenePriors)
- **요약**: 2D 감독 신호만으로 3D 장면 prior를 학습.

---

## 5. Asset & Texture Synthesis

### 5.1 3D Object Generation

#### ⭐ Objaverse-XL: A Universe of 10M+ 3D Objects
- **저자**: Matt Deitke et al. | **NeurIPS 2023** | 인용: ~500+
- **논문**: https://arxiv.org/abs/2307.05663
- **프로젝트**: https://objaverse.allenai.org/ | [GitHub](https://github.com/allenai/objaverse-xl)
- **요약**: 1000만 개 이상의 3D 오브젝트 데이터셋. 대규모 3D 생성 모델 학습의 기반 데이터.

---

### 5.2 Articulated Object Generation

#### Infinite Mobility: Scalable High-Fidelity Synthesis of Articulated Objects via Procedural Generation
- **저자**: (2025) | **RSS 2025**
- **논문**: https://arxiv.org/abs/2503.13424
- **프로젝트**: https://infinite-mobility.github.io/ | [GitHub](https://github.com/Intern-Nexus/Infinite-Mobility)
- **요약**: 절차적 생성을 통한 관절형 오브젝트의 확장 가능한 고충실도 합성.

#### Articulate-Anything: Automatic Modeling of Articulated Objects via a Vision-Language Foundation Model
- **저자**: (2024) | **ICLR 2025**
- **논문**: https://arxiv.org/abs/2410.13882
- **프로젝트**: https://articulate-anything.github.io/ | [GitHub](https://github.com/vlongle/articulate-anything)
- **요약**: VLM 기반 오픈 월드 3D 오브젝트의 자동 관절화.

#### MagicArticulate: Make Your 3D Models Articulation-Ready
- **저자**: (2025) | **CVPR 2025**
- **논문**: https://arxiv.org/abs/2502.12135
- **프로젝트**: https://chaoyuesong.github.io/MagicArticulate/ | [GitHub](https://github.com/Seed3D/MagicArticulate)
- **요약**: 3D 모델을 관절 준비 상태로 만드는 자동화 도구.

---

### 5.3 Scene Texture / Material

#### RoomPainter: View-Integrated Diffusion for Consistent Indoor Scene Texturing
- **저자**: (2025) | **CVPR 2025**
- **논문**: https://arxiv.org/abs/2412.16778
- **요약**: view-integrated diffusion으로 뷰 일관성 있는 실내 장면 텍스처링.

#### RoomTex: Texturing Compositional Indoor Scenes via Iterative Inpainting
- **저자**: (2024) | **ECCV 2024** | 인용: ~15+
- **논문**: https://arxiv.org/abs/2406.02461
- **프로젝트**: https://qwang666.github.io/RoomTex/ | [GitHub](https://github.com/qwang666/RoomTex-)
- **요약**: 반복적 inpainting으로 compositional 실내 장면 텍스처링.

#### MatFuse: Controllable Material Generation with Diffusion Models
- **저자**: (2024) | **CVPR 2024**
- **논문**: https://arxiv.org/abs/2308.11408
- **프로젝트**: https://gvecchio.com/matfuse/ | [GitHub](https://github.com/giuvecchio/matfuse-sd)
- **요약**: diffusion 기반 제어 가능한 재질 생성.

#### ControlMat: A Controlled Generative Approach to Material Capture
- **저자**: (2024) | **ACM TOG 2024**
- **논문**: https://arxiv.org/abs/2309.01700
- **프로젝트**: https://gvecchio.com/controlmat/
- **요약**: 생성적 접근법을 통한 제어 가능한 재질 캡처.

#### FlashTex: Fast Relightable Mesh Texturing with LightControlNet
- **저자**: (2024) | **ECCV 2024**
- **논문**: https://arxiv.org/abs/2402.13251
- **프로젝트**: https://flashtex.github.io/ | [GitHub](https://github.com/Roblox/FlashTex)
- **요약**: LightControlNet으로 빠른 재조명 가능 메시 텍스처링.

---

### 5.4 Scene Editing

#### BlenderAlchemy: Editing 3D Graphics with Vision-Language Models
- **저자**: (2024) | **ECCV 2024**
- **논문**: https://arxiv.org/abs/2404.17672
- **프로젝트**: https://ianhuang0630.github.io/BlenderAlchemyWeb/ | [GitHub](https://github.com/ianhuang0630/BlenderAlchemyOfficial)
- **요약**: VLM을 활용한 3D 그래픽 편집.

---

## 6. Applications

### 6.1 Embodied AI / Robotics

#### PhyScene: Physically Interactable 3D Scene Synthesis for Embodied AI
- **저자**: Yandan Yang et al. | **CVPR 2024 (Highlight)** | 인용: ~60+
- **논문**: https://arxiv.org/abs/2404.09465
- **프로젝트**: https://physcene.github.io/ | [GitHub](https://github.com/PhyScene/PhyScene)
- **요약**: Embodied AI가 상호작용 가능한 물리적으로 타당한 3D 실내 장면 합성.

#### SceneReVis: A Self-Reflective Vision-Grounded Framework for 3D Indoor Scene Synthesis via Multi-turn RL
- **저자**: (2026)
- **논문**: https://arxiv.org/abs/2602.09432
- **프로젝트**: https://scenerevis.github.io/ | [GitHub](https://github.com/Runder-sun/SceneReVis)
- **요약**: 다회전 강화학습 기반 자기반성 프레임워크로 공간 충돌 해결.

---

### 6.2 Human-Scene Interaction

#### HSI-GPT: A General-Purpose Large Scene-Motion-Language Model for Human Scene Interaction
- **저자**: Yuan Wang et al. | **CVPR 2025**
- **논문**: https://openaccess.thecvf.com/content/CVPR2025/html/Wang_HSI-GPT_A_General-Purpose_Large_Scene-Motion-Language_Model_for_Human_Scene_Interaction_CVPR_2025_paper.html
- **요약**: 장면-모션-언어 통합 대규모 모델로 인간-장면 상호작용 이해.

---

### 6.3 Layout Editing & Rearrangement

#### LayoutTransformer: Layout Generation and Completion with Self-attention
- **저자**: (2021) | **ICCV 2021**
- **논문**: https://arxiv.org/abs/2006.14615
- **프로젝트**: https://kampta.github.io/layout/ | [GitHub](https://github.com/kampta/DeepLayout)
- **요약**: Self-attention 기반 레이아웃 생성 및 완성. 범용 레이아웃(문서·UI·3D 포함) 생성의 기반 연구.

#### LEGO-Net: Learning Regular Rearrangements of Objects in Rooms
- **저자**: Qiuhong Anna Wei et al. | **CVPR 2023** | 인용: ~50+
- **논문**: https://arxiv.org/abs/2301.09629
- **프로젝트**: https://ivl.cs.brown.edu/projects/lego-net
- **요약**: 물리적·미적 규칙을 학습해 오브젝트를 재배치하는 모델. 기존 장면 최적화에 집중.

---

## 7. Datasets & Evaluation

### 7.1 Indoor Scene Datasets

#### ⭐ ScanNet: Richly-Annotated 3D Reconstructions of Indoor Scenes
- **저자**: Angela Dai et al. | **CVPR 2017** | 인용: **~9,000+**
- **논문**: https://arxiv.org/abs/1702.04405
- **프로젝트**: http://www.scan-net.org/ | [GitHub](https://github.com/ScanNet/ScanNet)
- **요약**: 1513개 실내 장면의 RGB-D 스캔 + 의미론적 annotation. 3D indoor understanding의 표준 데이터셋.

#### ⭐ 3D-FRONT: 3D Furnished Rooms with layOuts and semaNTics
- **저자**: Huan Fu et al. | **ICCV 2021** | 인용: ~600+
- **논문**: https://arxiv.org/abs/2011.09127
- **프로젝트**: [GitHub](https://github.com/3D-FRONT-FUTURE/3D-FUTURE-ToolBox)
- **요약**: 전문 인테리어 디자이너가 설계한 18,000여 개의 실내 장면. Layout 생성 연구의 핵심 데이터셋.

#### ⭐ ScanNet++: A High-Fidelity Dataset of 3D Indoor Scenes
- **저자**: Chandan Yeshwanth et al. | **ICCV 2023 (Oral)** | 인용: **577**
- **논문**: https://arxiv.org/abs/2308.11417
- **프로젝트**: [GitHub](https://github.com/scannetpp/scannetpp)
- **요약**: ScanNet 대비 훨씬 고해상도 3D 실내 장면 데이터셋 (DSLR + LiDAR).

#### Habitat Synthetic Scenes Dataset (HSSD-200)
- **저자**: (2024) | **CVPR 2024** | 인용: ~100+
- **논문**: https://arxiv.org/abs/2306.11290
- **프로젝트**: https://3dlg-hcvc.github.io/hssd/ | [GitHub](https://github.com/3dlg-hcvc/hssd)
- **요약**: Embodied AI용 고품질 합성 실내 장면 데이터셋. 211개의 photorealistic 씬.

#### HM3D: Habitat-Matterport 3D Dataset
- **저자**: (2021) | **NeurIPS 2021** | 인용: ~400+
- **논문**: https://arxiv.org/abs/2109.08238
- **프로젝트**: https://aihabitat.org/datasets/hm3d/ | [GitHub](https://github.com/facebookresearch/habitat-matterport3d-dataset)
- **요약**: 1000개 대규모 실내 환경. Embodied AI navigation의 주요 benchmark.

#### 3D-FUTURE: 3D Furniture shape with TextURE
- **저자**: (2021) | **IJCV 2021** | 인용: ~400+
- **논문**: https://arxiv.org/abs/2009.09633
- **프로젝트**: [GitHub](https://github.com/3D-FRONT-FUTURE/3D-FUTURE-ToolBox)
- **요약**: 9,992개 가구 3D 메시 + 텍스처. 3D-FRONT와 함께 사용되는 오브젝트 라이브러리.

---

### 7.2 Evaluation Metrics

#### SceneEval: Evaluating Semantic Coherence in Text-Conditioned 3D Indoor Scene Synthesis
- **저자**: (2025) | 인용: 7
- **논문**: https://arxiv.org/abs/2503.14756
- **프로젝트**: https://3dlg-hcvc.github.io/SceneEval/ | [GitHub](https://github.com/3dlg-hcvc/SceneEval)
- **요약**: 텍스트 조건부 3D 실내 장면 합성의 의미론적 일관성 평가 프레임워크.

#### RelScene: A Benchmark and Baseline for Spatial Relations in Text-driven 3D Scene Generation
- **저자**: (2024) | **ACM MM 2024** | 인용: 4
- **논문**: https://openreview.net/forum?id=GIw7pmMPPX
- **요약**: 텍스트 기반 3D 장면 생성에서 공간 관계 평가 benchmark.

**일반적으로 사용되는 메트릭:**
| 메트릭 | 설명 | 사용 분야 |
|--------|------|-----------|
| FID (Fréchet Inception Distance) | 생성 품질 | Layout / Image |
| KL-divergence | 분포 유사성 | Layout |
| Collision Rate | 물리적 타당성 | 3D Scene |
| Out-of-boundary Rate | 공간 제약 만족 여부 | 3D Scene |
| CLIP Score | 텍스트-이미지 정합성 | Text-to-Scene |
| PSNR / SSIM / LPIPS | 렌더링 품질 | NeRF, 3DGS |

---

## 분야별 핵심 논문 요약

| 카테고리 | 기초 논문 ⭐ | 인용 수 | 의의 |
|----------|------------|---------|------|
| Scene Reconstruction | NeRF (ECCV 2020) | ~17,000 | 3D 표현 패러다임 전환 |
| Scene Reconstruction | 3DGS (SIGGRAPH 2023) | ~3,000 | 실시간 렌더링 가능 |
| Dataset | ScanNet (CVPR 2017) | ~9,000 | 3D indoor 연구 표준 데이터 |
| Dataset | 3D-FRONT (ICCV 2021) | ~600 | Layout 생성 학습의 표준 |
| Dataset | ScanNet++ (ICCV 2023) | 577 | 고해상도 benchmark 표준 |
| Layout (CNN) | Deep Conv Priors (SIGGRAPH 2018) | 261 | DL 기반 scene synthesis 선구 |
| Layout (Transformer) | ATISS (NeurIPS 2021) | 171 | Transformer 기반 layout 생성 표준 |
| Layout (Transformer) | SceneFormer (3DV 2021) | 164 | Scene graph + Transformer 결합 |
| Layout (Diffusion) | DiffuScene (CVPR 2024) | 119 | Diffusion 기반 scene synthesis 표준 |
| Procedural | Infinigen Indoors (CVPR 2024) | 102 | 포토리얼리스틱 절차적 생성 |
| LLM-based | Holodeck (CVPR 2024) | ~100 | LLM 기반 장면 생성 대표작 |
| Graph+Diffusion | CommonScenes (NeurIPS 2023) | ~60 | Scene graph + Diffusion 결합 선구 |
| Instruction | InstructScene (ICLR 2024) | 80 | Instruction 기반 scene 생성 |
